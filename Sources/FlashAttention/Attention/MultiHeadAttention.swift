//
//  MultiHeadAttention.swift
//  FlashAttention
//
//  Created by bghira on 9/15/24.
//

import Metal

/// Multi-head flash attention implementation with optimized broadcast semantics
public class MultiHeadAttention {
  private let device: MTLDevice
  private let commandQueue: MTLCommandQueue
  private var pipelineCache: [String: MTLComputePipelineState] = [:]

  public init(device: MTLDevice) {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      fatalError("Could not create Metal command queue")
    }
    commandQueue = queue
  }

  /// Perform multi-head attention forward pass
  /// - Parameters:
  ///   - query: Query tensor buffer [B, H, S_q, D] or compatible broadcast shape
  ///   - key: Key tensor buffer [B, H_kv, S_k, D] or compatible broadcast shape
  ///   - value: Value tensor buffer [B, H_kv, S_k, D] or compatible broadcast shape
  ///   - output: Output tensor buffer [B, H, S_q, D]
  ///   - logsumexp: Logsumexp output buffer [B, H, S_q] (optional)
  ///   - descriptor: Multi-head attention configuration
  /// - Returns: Command buffer for execution
  public func forward(
    query: MTLBuffer,
    key: MTLBuffer,
    value: MTLBuffer,
    output: MTLBuffer,
    logsumexp: MTLBuffer? = nil,
    descriptor: MultiHeadAttentionDescriptor
  ) -> MTLCommandBuffer? {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      print("Error: Failed to create command buffer")
      return nil
    }

    switch descriptor.dispatchStrategy {
    case .perBatchHead:
      return dispatchPerBatchHead(
        commandBuffer: commandBuffer,
        query: query, key: key, value: value, output: output,
        logsumexp: logsumexp, descriptor: descriptor
      )

    case .perBatch:
      return dispatchPerBatch(
        commandBuffer: commandBuffer,
        query: query, key: key, value: value, output: output,
        logsumexp: logsumexp, descriptor: descriptor
      )

    case .batched, .auto:
      return dispatchBatched(
        commandBuffer: commandBuffer,
        query: query, key: key, value: value, output: output,
        logsumexp: logsumexp, descriptor: descriptor
      )
    }
  }

  /// Dispatch strategy: one kernel per (batch, head) pair - maximum parallelism
  private func dispatchPerBatchHead(
    commandBuffer: MTLCommandBuffer,
    query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, output: MTLBuffer,
    logsumexp: MTLBuffer?, descriptor: MultiHeadAttentionDescriptor
  ) -> MTLCommandBuffer? {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    let kernelDescriptors = descriptor.kernelDescriptors(type: .forward)

    for batchIdx in 0..<descriptor.queryShape.batchSize {
      for headIdx in 0..<descriptor.queryShape.numHeads {
        let kernelDesc = kernelDescriptors[Int(batchIdx * descriptor.queryShape.numHeads + headIdx)]
        let kernel = AttentionKernel(descriptor: kernelDesc)

        guard let pipelineState = getOrCreatePipelineState(for: kernel, descriptor: descriptor) else {
          print("Error: Failed to create pipeline state for batch \(batchIdx), head \(headIdx)")
          continue
        }

        encoder.setComputePipelineState(pipelineState)

        // Calculate buffer offsets for this batch/head
        let offsets = calculateBufferOffsets(
          batchIndex: batchIdx, headIndex: headIdx, descriptor: descriptor
        )

        // Set buffers with offsets
        encoder.setBuffer(query, offset: offsets.query, index: 0)
        encoder.setBuffer(key, offset: offsets.key, index: 1)
        encoder.setBuffer(value, offset: offsets.value, index: 2)
        encoder.setBuffer(output, offset: offsets.output, index: 3)

        if let logsumexp = logsumexp {
          encoder.setBuffer(logsumexp, offset: offsets.logsumexp, index: 4)
        }

        // Set threadgroup memory
        encoder.setThreadgroupMemoryLength(
          Int(kernel.threadgroupMemoryAllocation), index: 0
        )

        // Dispatch this head
        let blockCount = ceilDivide(
          Int(descriptor.queryShape.sequenceLength),
          Int(kernel.blockDimensions.parallelization)
        )
        let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
        let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
      }
    }

    encoder.endEncoding()
    return commandBuffer
  }

  /// Dispatch strategy: one kernel per batch - heads processed within kernel
  private func dispatchPerBatch(
    commandBuffer: MTLCommandBuffer,
    query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, output: MTLBuffer,
    logsumexp: MTLBuffer?, descriptor: MultiHeadAttentionDescriptor
  ) -> MTLCommandBuffer? {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    let kernelDescriptors = descriptor.kernelDescriptors(type: .forward)

    for batchIdx in 0..<descriptor.queryShape.batchSize {
      let kernelDesc = kernelDescriptors[Int(batchIdx)]
      let kernel = AttentionKernel(descriptor: kernelDesc)

      guard let pipelineState = getOrCreateMultiHeadPipelineState(
        for: kernel, descriptor: descriptor, processingMode: .perBatch
      ) else {
        print("Error: Failed to create multi-head pipeline state for batch \(batchIdx)")
        continue
      }

      encoder.setComputePipelineState(pipelineState)

      // Calculate buffer offsets for this batch
      let offsets = calculateBatchOffsets(batchIndex: batchIdx, descriptor: descriptor)

      // Set buffers with batch offsets
      encoder.setBuffer(query, offset: offsets.query, index: 0)
      encoder.setBuffer(key, offset: offsets.key, index: 1)
      encoder.setBuffer(value, offset: offsets.value, index: 2)
      encoder.setBuffer(output, offset: offsets.output, index: 3)

      if let logsumexp = logsumexp {
        encoder.setBuffer(logsumexp, offset: offsets.logsumexp, index: 4)
      }

      // Set multi-head parameters
      var numHeads = descriptor.queryShape.numHeads
      var numKVHeads = descriptor.keyShape.numHeads
      encoder.setBytes(&numHeads, length: MemoryLayout<UInt32>.size, index: 5)
      encoder.setBytes(&numKVHeads, length: MemoryLayout<UInt32>.size, index: 6)

      // Set threadgroup memory
      encoder.setThreadgroupMemoryLength(
        Int(kernel.threadgroupMemoryAllocation), index: 0
      )

      // Dispatch with head dimension included in grid
      let blockCount = ceilDivide(
        Int(descriptor.queryShape.sequenceLength),
        Int(kernel.blockDimensions.parallelization)
      )
      let gridSize = MTLSize(
        width: blockCount,
        height: Int(descriptor.queryShape.numHeads),
        depth: 1
      )
      let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

      encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    }

    encoder.endEncoding()
    return commandBuffer
  }

  /// Dispatch strategy: single kernel for entire batch - maximum batching
  private func dispatchBatched(
    commandBuffer: MTLCommandBuffer,
    query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, output: MTLBuffer,
    logsumexp: MTLBuffer?, descriptor: MultiHeadAttentionDescriptor
  ) -> MTLCommandBuffer? {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    let kernelDescriptors = descriptor.kernelDescriptors(type: .forward)
    let kernelDesc = kernelDescriptors[0]
    let kernel = AttentionKernel(descriptor: kernelDesc)

    guard let pipelineState = getOrCreateMultiHeadPipelineState(
      for: kernel, descriptor: descriptor, processingMode: .batched
    ) else {
      print("Error: Failed to create batched multi-head pipeline state")
      return nil
    }

    encoder.setComputePipelineState(pipelineState)

    // Set buffers (no offsets for batched mode)
    encoder.setBuffer(query, offset: 0, index: 0)
    encoder.setBuffer(key, offset: 0, index: 1)
    encoder.setBuffer(value, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)

    if let logsumexp = logsumexp {
      encoder.setBuffer(logsumexp, offset: 0, index: 4)
    }

    // Set batch and head parameters
    var batchSize = descriptor.queryShape.batchSize
    var numHeads = descriptor.queryShape.numHeads
    var numKVHeads = descriptor.keyShape.numHeads
    var sequenceLength = descriptor.queryShape.sequenceLength
    var kvSequenceLength = descriptor.keyShape.sequenceLength
    var headDimension = descriptor.queryShape.headDimension

    encoder.setBytes(&batchSize, length: MemoryLayout<UInt32>.size, index: 5)
    encoder.setBytes(&numHeads, length: MemoryLayout<UInt32>.size, index: 6)
    encoder.setBytes(&numKVHeads, length: MemoryLayout<UInt32>.size, index: 7)
    encoder.setBytes(&sequenceLength, length: MemoryLayout<UInt32>.size, index: 8)
    encoder.setBytes(&kvSequenceLength, length: MemoryLayout<UInt32>.size, index: 9)
    encoder.setBytes(&headDimension, length: MemoryLayout<UInt16>.size, index: 10)

    // Set broadcast mode
    var broadcastModeRaw = encodeBroadcastMode(descriptor.broadcastMode)
    encoder.setBytes(&broadcastModeRaw, length: MemoryLayout<UInt32>.size, index: 11)

    // Set threadgroup memory
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0
    )

    // Dispatch with full batch and head dimensions
    let blockCount = ceilDivide(
      Int(descriptor.queryShape.sequenceLength),
      Int(kernel.blockDimensions.parallelization)
    )
    let gridSize = MTLSize(
      width: blockCount,
      height: Int(descriptor.queryShape.numHeads),
      depth: Int(descriptor.queryShape.batchSize)
    )
    let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()

    return commandBuffer
  }

  // MARK: - Helper Methods

  private func ceilDivide(_ numerator: Int, _ denominator: Int) -> Int {
    (numerator + denominator - 1) / denominator
  }

  private func encodeBroadcastMode(_ mode: MultiHeadBroadcastMode) -> UInt32 {
    switch mode {
    case .standard: return 0
    case .groupedQuery: return 1
    case .multiQuery: return 2
    case .crossAttention: return 3
    case .custom: return 4
    }
  }

  private enum MultiHeadProcessingMode {
    case perBatch
    case batched
  }

  private func getOrCreatePipelineState(
    for kernel: AttentionKernel, descriptor: MultiHeadAttentionDescriptor
  ) -> MTLComputePipelineState? {
    let source = kernel.createSource()
    let cacheKey = String(source.hashValue)

    if let cached = pipelineCache[cacheKey] {
      return cached
    }

    do {
      let library = try device.makeLibrary(source: source, options: nil)

      let functionConstants = MTLFunctionConstantValues()
      descriptor.baseDescriptor.setFunctionConstants(functionConstants)

      let function = try library.makeFunction(name: "attention", constantValues: functionConstants)
      let pipelineState = try device.makeComputePipelineState(function: function)

      pipelineCache[cacheKey] = pipelineState
      return pipelineState
    } catch {
      print("Pipeline creation error: \(error)")
      return nil
    }
  }

  private func getOrCreateMultiHeadPipelineState(
    for kernel: AttentionKernel, descriptor: MultiHeadAttentionDescriptor,
    processingMode: MultiHeadProcessingMode
  ) -> MTLComputePipelineState? {
    // For now, use single-head kernels with offset calculations
    // Future: implement dedicated multi-head kernels
    return getOrCreatePipelineState(for: kernel, descriptor: descriptor)
  }

  private struct BufferOffsets {
    let query: Int
    let key: Int
    let value: Int
    let output: Int
    let logsumexp: Int
  }

  private func calculateBufferOffsets(
    batchIndex: UInt32, headIndex: UInt32, descriptor: MultiHeadAttentionDescriptor
  ) -> BufferOffsets {
    let qShape = descriptor.queryShape
    let kShape = descriptor.keyShape
    let vShape = descriptor.valueShape

    // Calculate offsets based on memory layout [B, H, S, D]
    let qBatchStride = Int(qShape.numHeads * qShape.sequenceLength * UInt32(qShape.headDimension))
    let qHeadStride = Int(qShape.sequenceLength * UInt32(qShape.headDimension))

    let kBatchStride = Int(kShape.numHeads * kShape.sequenceLength * UInt32(kShape.headDimension))
    let vBatchStride = Int(vShape.numHeads * vShape.sequenceLength * UInt32(vShape.headDimension))

    // Handle broadcast modes for K/V head indices
    let kvHeadIndex: UInt32
    switch descriptor.broadcastMode {
    case .standard, .crossAttention:
      kvHeadIndex = headIndex
    case .groupedQuery(let numKVHeads):
      kvHeadIndex = headIndex % numKVHeads
    case .multiQuery:
      kvHeadIndex = 0
    case .custom:
      kvHeadIndex = headIndex // Simplified for custom mode
    }

    let kHeadStride = Int(kShape.sequenceLength * UInt32(kShape.headDimension))
    let vHeadStride = Int(vShape.sequenceLength * UInt32(vShape.headDimension))

    // Element size in bytes (assuming FP16 for now)
    let elementSize = 2

    return BufferOffsets(
      query: (Int(batchIndex) * qBatchStride + Int(headIndex) * qHeadStride) * elementSize,
      key: (Int(batchIndex) * kBatchStride + Int(kvHeadIndex) * kHeadStride) * elementSize,
      value: (Int(batchIndex) * vBatchStride + Int(kvHeadIndex) * vHeadStride) * elementSize,
      output: (Int(batchIndex) * qBatchStride + Int(headIndex) * qHeadStride) * elementSize,
      logsumexp: (Int(batchIndex) * Int(qShape.numHeads * qShape.sequenceLength) + Int(headIndex * qShape.sequenceLength)) * 4 // FP32
    )
  }

  private func calculateBatchOffsets(
    batchIndex: UInt32, descriptor: MultiHeadAttentionDescriptor
  ) -> BufferOffsets {
    let qShape = descriptor.queryShape
    let kShape = descriptor.keyShape
    let vShape = descriptor.valueShape

    let qBatchStride = Int(qShape.numHeads * qShape.sequenceLength * UInt32(qShape.headDimension))
    let kBatchStride = Int(kShape.numHeads * kShape.sequenceLength * UInt32(kShape.headDimension))
    let vBatchStride = Int(vShape.numHeads * vShape.sequenceLength * UInt32(vShape.headDimension))

    // Element size in bytes (assuming FP16 for now)
    let elementSize = 2

    return BufferOffsets(
      query: Int(batchIndex) * qBatchStride * elementSize,
      key: Int(batchIndex) * kBatchStride * elementSize,
      value: Int(batchIndex) * vBatchStride * elementSize,
      output: Int(batchIndex) * qBatchStride * elementSize,
      logsumexp: Int(batchIndex) * Int(qShape.numHeads * qShape.sequenceLength) * 4 // FP32
    )
  }
}
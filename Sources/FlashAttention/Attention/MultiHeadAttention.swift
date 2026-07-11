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
    descriptor: MultiHeadAttentionDescriptor,
    maskBuffer: MTLBuffer? = nil
  )
    -> MTLCommandBuffer?
  {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      print("Error: Failed to create command buffer")
      return nil
    }

    var resolvedDescriptor = descriptor
    if var sparseMask = resolvedDescriptor.baseDescriptor.sparseMask {
      sparseMask.isMQA = resolvedDescriptor.broadcastMode.isMultiQuery
      sparseMask.numKVHeads = resolvedDescriptor.keyShape.numHeads
      resolvedDescriptor.baseDescriptor.sparseMask = sparseMask
    }

    let resolvedMaskBuffer = maskBuffer ?? resolvedDescriptor.baseDescriptor.sparseMask?.maskBuffer

    switch resolvedDescriptor.dispatchStrategy {
    case .perBatchHead:
      return dispatchPerBatchHead(
        commandBuffer: commandBuffer,
        query: query, key: key, value: value, output: output,
        logsumexp: logsumexp, descriptor: resolvedDescriptor,
        maskBuffer: resolvedMaskBuffer
      )

    case .perBatch:
      return dispatchPerBatch(
        commandBuffer: commandBuffer,
        query: query, key: key, value: value, output: output,
        logsumexp: logsumexp, descriptor: resolvedDescriptor,
        maskBuffer: resolvedMaskBuffer
      )

    case .batched, .auto:
      return dispatchBatched(
        commandBuffer: commandBuffer,
        query: query, key: key, value: value, output: output,
        logsumexp: logsumexp, descriptor: resolvedDescriptor,
        maskBuffer: resolvedMaskBuffer
      )
    }
  }

  /// Dispatch strategy: parallel dispatch for all (batch, head) pairs using 3D grid
  private func dispatchPerBatchHead(
    commandBuffer: MTLCommandBuffer,
    query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, output: MTLBuffer,
    logsumexp: MTLBuffer?, descriptor: MultiHeadAttentionDescriptor,
    maskBuffer: MTLBuffer?
  )
    -> MTLCommandBuffer?
  {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    // Use first kernel descriptor as representative (they should all be similar)
    let kernelDescriptors = descriptor.kernelDescriptors(type: .forward)
    let kernelDesc = kernelDescriptors[0]
    let kernel = AttentionKernel(descriptor: kernelDesc)

    guard let pipelineState = getOrCreatePipelineState(for: kernel, descriptor: descriptor) else {
      print("Error: Failed to create pipeline state for parallel dispatch")
      return nil
    }

    encoder.setComputePipelineState(pipelineState)

    // Set buffers without offsets (kernel handles offsets internally)
    encoder.setBuffer(query, offset: 0, index: 0)
    encoder.setBuffer(key, offset: 0, index: 1)
    encoder.setBuffer(value, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)

    if let logsumexp {
      encoder.setBuffer(logsumexp, offset: 0, index: 4)
    }

    // Calculate buffer index after quantization parameters
    let baseIndex = 5 // After standard buffers
    let quantBindings = quantizationBindings(for: descriptor)
    var bufferIndex = baseIndex
    for binding in quantBindings {
      var scale = binding.parameters.scale
      var zeroPoint = binding.parameters.zeroPoint
      var strategy = UInt32(binding.parameters.strategy.rawValue)
      var strategyVersion = UInt32(binding.parameters.strategyVersion)

      encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: bufferIndex)
      bufferIndex += 1

      encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: bufferIndex)
      bufferIndex += 1

      encoder.setBytes(&strategy, length: MemoryLayout<UInt32>.size, index: bufferIndex)
      bufferIndex += 1

      encoder.setBytes(&strategyVersion, length: MemoryLayout<UInt32>.size, index: bufferIndex)
      bufferIndex += 1
    }

    let multiHeadParamIndex = bufferIndex

    // Set multi-head parameters
    var numHeads = descriptor.queryShape.numHeads
    var numKVHeads = descriptor.keyShape.numHeads
    var headDimension = UInt32(descriptor.queryShape.headDimension)
    var sequenceLength = descriptor.queryShape.sequenceLength

    encoder.setBytes(&numHeads, length: MemoryLayout<UInt32>.size, index: multiHeadParamIndex)
    encoder.setBytes(&numKVHeads, length: MemoryLayout<UInt32>.size, index: multiHeadParamIndex + 1)
    encoder.setBytes(
      &headDimension,
      length: MemoryLayout<UInt32>.size,
      index: multiHeadParamIndex + 2
    )
    encoder.setBytes(
      &sequenceLength,
      length: MemoryLayout<UInt32>.size,
      index: multiHeadParamIndex + 3
    )

    if let maskBuffer {
      encoder.setBuffer(maskBuffer, offset: 0, index: multiHeadParamIndex + 4)
    } else {
      encoder.setBuffer(nil, offset: 0, index: multiHeadParamIndex + 4)
    }

    // Set threadgroup memory
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0
    )

    // Dispatch with 3D grid for all heads and batches in parallel
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

  /// Dispatch strategy: unified dispatch with all batches and heads in parallel
  private func dispatchPerBatch(
    commandBuffer: MTLCommandBuffer,
    query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, output: MTLBuffer,
    logsumexp: MTLBuffer?, descriptor: MultiHeadAttentionDescriptor,
    maskBuffer: MTLBuffer?
  )
    -> MTLCommandBuffer?
  {
    // Use the same implementation as dispatchBatched since we're now doing parallel dispatch
    dispatchBatched(
      commandBuffer: commandBuffer,
      query: query, key: key, value: value, output: output,
      logsumexp: logsumexp, descriptor: descriptor,
      maskBuffer: maskBuffer
    )
  }

  /// Dispatch strategy: single kernel for entire batch - maximum batching
  private func dispatchBatched(
    commandBuffer: MTLCommandBuffer,
    query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, output: MTLBuffer,
    logsumexp: MTLBuffer?, descriptor: MultiHeadAttentionDescriptor,
    maskBuffer: MTLBuffer?
  )
    -> MTLCommandBuffer?
  {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    let kernelDescriptors = descriptor.kernelDescriptors(type: .forward)
    let kernelDesc = kernelDescriptors[0]
    let kernel = AttentionKernel(descriptor: kernelDesc)

    guard
      let pipelineState = getOrCreateMultiHeadPipelineState(
        for: kernel, descriptor: descriptor, processingMode: .batched
      )
    else {
      print("Error: Failed to create batched multi-head pipeline state")
      return nil
    }

    encoder.setComputePipelineState(pipelineState)

    // Bind operands in the exact slot order the generated kernel expects
    // (see AttentionKernel.createBufferBindings): Q@0 K@1 V@2 O@3 L@4,
    // then (for non-quantized kernels) Q_strides@5 K_strides@6 V_strides@7,
    // then num_heads@8 num_kv_heads@9 head_dim@10 seq@11, then mask@12.
    encoder.setBuffer(query, offset: 0, index: 0)
    encoder.setBuffer(key, offset: 0, index: 1)
    encoder.setBuffer(value, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)

    // The kernel always writes L (logsumexp). Provide a scratch buffer for
    // forward-only calls that pass logsumexp=nil, otherwise the kernel writes
    // to an unbound slot and corrupts output.
    let lBuffer: MTLBuffer
    if let logsumexp {
      lBuffer = logsumexp
    } else {
      let lCount = Int(descriptor.queryShape.batchSize)
        * Int(descriptor.queryShape.numHeads)
        * Int(descriptor.queryShape.sequenceLength)
      // storageModeShared + explicit zero-fill: the forward kernel seeds its
      // online-softmax running max from L, so an uninitialised (garbage) L
      // corrupts the FIRST dispatch of a shape (subsequent dispatches see the
      // value written by the first and look correct). Zeroing removes that
      // cold-dispatch hazard.
      guard
        let zeroed = device.makeBuffer(
          length: max(lCount, 1) * MemoryLayout<UInt32>.size,
          options: .storageModeShared
        )
      else {
        return nil
      }
      memset(zeroed.contents(), 0, max(lCount, 1) * MemoryLayout<UInt32>.size)
      lBuffer = zeroed
    }
    encoder.setBuffer(lBuffer, offset: 0, index: 4)

    // Strides @5/6/7: bind nil so the kernel uses its contiguous-fallback
    // offset math (batch*num_heads + head) * seq * dim, which matches the
    // contiguous BHSD layout the host now passes. (The kernel's stride-index
    // path reads head stride from the wrong array slot; the contiguous path
    // is correct, so disable the stride path by passing null.)
    encoder.setBuffer(nil, offset: 0, index: 5)
    encoder.setBuffer(nil, offset: 0, index: 6)
    encoder.setBuffer(nil, offset: 0, index: 7)

    // Multi-head parameters @8..11
    var numHeads = descriptor.queryShape.numHeads
    var numKVHeads = descriptor.keyShape.numHeads
    var headDimension = UInt32(descriptor.queryShape.headDimension)
    var sequenceLength = descriptor.queryShape.sequenceLength
    encoder.setBytes(&numHeads, length: MemoryLayout<UInt32>.size, index: 8)
    encoder.setBytes(&numKVHeads, length: MemoryLayout<UInt32>.size, index: 9)
    encoder.setBytes(
      &headDimension,
      length: MemoryLayout<UInt32>.size,
      index: 10
    )
    encoder.setBytes(
      &sequenceLength,
      length: MemoryLayout<UInt32>.size,
      index: 11
    )

    // Mask @12
    if let maskBuffer {
      encoder.setBuffer(maskBuffer, offset: 0, index: 12)
    } else {
      encoder.setBuffer(nil, offset: 0, index: 12)
    }

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
    case .standard: 0
    case .groupedQuery: 1
    case .multiQuery: 2
    case .crossAttention: 3
    case .custom: 4
    }
  }

  private func getQuantizedOperandCount(_ descriptor: MultiHeadAttentionDescriptor) -> Int {
    quantizationBindings(for: descriptor).count
  }

  public struct QuantizationBinding {
    public let operand: AttentionOperand
    public let parameters: QuantizationParameters
  }

  public func quantizationBindings(for descriptor: MultiHeadAttentionDescriptor)
    -> [QuantizationBinding]
  {
    let orderedOperands: [AttentionOperand] = [.Q, .K, .V, .O]
    return orderedOperands.compactMap { operand in
      guard let params = descriptor.quantizationParameters[operand] else {
        return nil
      }
      return QuantizationBinding(operand: operand, parameters: params)
    }
  }

  private enum MultiHeadProcessingMode {
    case perBatch
    case batched
  }

  private func getOrCreatePipelineState(
    for kernel: AttentionKernel, descriptor: MultiHeadAttentionDescriptor
  )
    -> MTLComputePipelineState?
  {
    let source = kernel.createSource()
    let cacheKey = String(source.hashValue)

    if let cached = pipelineCache[cacheKey] {
      return cached
    }

    do {
      let patchedSource = GEMMBFloatHeaderEmbedder.embed(into: source)
      let opts = MTLCompileOptions()
      opts.languageVersion = .version3_2
      let library = try device.makeLibrary(source: patchedSource, options: opts)

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
    processingMode _: MultiHeadProcessingMode
  )
    -> MTLComputePipelineState?
  {
    // For now, use single-head kernels with offset calculations
    // Future: implement dedicated multi-head kernels
    getOrCreatePipelineState(for: kernel, descriptor: descriptor)
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
  )
    -> BufferOffsets
  {
    let qShape = descriptor.queryShape
    let kShape = descriptor.keyShape
    let vShape = descriptor.valueShape

    // Calculate offsets based on memory layout [B, H, S, D]
    let qBatchStride = Int(qShape.numHeads * qShape.sequenceLength * UInt32(qShape.headDimension))
    let qHeadStride = Int(qShape.sequenceLength * UInt32(qShape.headDimension))

    let kBatchStride = Int(kShape.numHeads * kShape.sequenceLength * UInt32(kShape.headDimension))
    let vBatchStride = Int(vShape.numHeads * vShape.sequenceLength * UInt32(vShape.headDimension))

    // Handle broadcast modes for K/V head indices
    let kvHeadIndex: UInt32 = switch descriptor.broadcastMode {
    case .standard, .crossAttention:
      headIndex
    case let .groupedQuery(numKVHeads):
      headIndex % numKVHeads
    case .multiQuery:
      0
    case .custom:
      headIndex // Simplified for custom mode
    }

    let kHeadStride = Int(kShape.sequenceLength * UInt32(kShape.headDimension))
    let vHeadStride = Int(vShape.sequenceLength * UInt32(vShape.headDimension))

    // Element size in bytes derived from descriptor precision
    let elementSize: Int = descriptor.baseDescriptor.lowPrecisionInputs ? MemoryLayout<Float16>.stride :
      MemoryLayout<Float>.stride

    return BufferOffsets(
      query: (Int(batchIndex) * qBatchStride + Int(headIndex) * qHeadStride) * elementSize,
      key: (Int(batchIndex) * kBatchStride + Int(kvHeadIndex) * kHeadStride) * elementSize,
      value: (Int(batchIndex) * vBatchStride + Int(kvHeadIndex) * vHeadStride) * elementSize,
      output: (Int(batchIndex) * qBatchStride + Int(headIndex) * qHeadStride) * elementSize,
      logsumexp: (
        Int(batchIndex) * Int(qShape.numHeads * qShape.sequenceLength) +
          Int(headIndex * qShape.sequenceLength)
      ) * 4 // FP32
    )
  }

  private func calculateBatchOffsets(
    batchIndex: UInt32, descriptor: MultiHeadAttentionDescriptor
  )
    -> BufferOffsets
  {
    let qShape = descriptor.queryShape
    let kShape = descriptor.keyShape
    let vShape = descriptor.valueShape

    let qBatchStride = Int(qShape.numHeads * qShape.sequenceLength * UInt32(qShape.headDimension))
    let kBatchStride = Int(kShape.numHeads * kShape.sequenceLength * UInt32(kShape.headDimension))
    let vBatchStride = Int(vShape.numHeads * vShape.sequenceLength * UInt32(vShape.headDimension))

    // Element size in bytes derived from descriptor precision
    let elementSize: Int = descriptor.baseDescriptor.lowPrecisionInputs ? MemoryLayout<Float16>.stride :
      MemoryLayout<Float>.stride

    return BufferOffsets(
      query: Int(batchIndex) * qBatchStride * elementSize,
      key: Int(batchIndex) * kBatchStride * elementSize,
      value: Int(batchIndex) * vBatchStride * elementSize,
      output: Int(batchIndex) * qBatchStride * elementSize,
      logsumexp: Int(batchIndex) * Int(qShape.numHeads * qShape.sequenceLength) * 4 // FP32
    )
  }

  // MARK: - Backward

  /// Backward pass for flash attention.
  ///
  /// Dispatches `.backwardQuery` (computes D + dQ) then `.backwardKeyValue`
  /// (computes dK + dV, depends on D) in a single command buffer.
  public func backward(
    query: MTLBuffer,
    key: MTLBuffer,
    value: MTLBuffer,
    output: MTLBuffer,
    dOutput: MTLBuffer,
    logsumexp: MTLBuffer,
    dQuery: MTLBuffer,
    dKey: MTLBuffer,
    dValue: MTLBuffer,
    dBuffer: MTLBuffer,
    descriptor: MultiHeadAttentionDescriptor,
    maskBuffer: MTLBuffer? = nil
  )
    -> MTLCommandBuffer?
  {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      return nil
    }

    // --- Phase 1: backwardQuery (computes D intermediate + dQ) ---
    // Buffer bindings (from AttentionKernel.createBufferBindings):
    // Q@0 K@1 V@2 O@3 L@4 D@5 dO@6 dQ@9
    // nil strides @10 @11 @12
    // num_heads@13 num_kv_heads@14 head_dim@15 seq_len@16
    // nil mask@17
    let bqKernelDescs = descriptor.kernelDescriptors(type: .backwardQuery)
    let bqKernel = AttentionKernel(descriptor: bqKernelDescs[0])

    guard
      let bqPipeline = getOrCreatePipelineState(for: bqKernel, descriptor: descriptor),
      let bqEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      return nil
    }

    let bqOpts = MTLCompileOptions()
    bqOpts.languageVersion = .version3_2
    _ = bqOpts

    bqEncoder.setComputePipelineState(bqPipeline)
    bqEncoder.setBuffer(query, offset: 0, index: 0)
    bqEncoder.setBuffer(key, offset: 0, index: 1)
    bqEncoder.setBuffer(value, offset: 0, index: 2)
    bqEncoder.setBuffer(output, offset: 0, index: 3)
    bqEncoder.setBuffer(logsumexp, offset: 0, index: 4)
    bqEncoder.setBuffer(dBuffer, offset: 0, index: 5)
    bqEncoder.setBuffer(dOutput, offset: 0, index: 6)
    bqEncoder.setBuffer(dQuery, offset: 0, index: 9)
    // nil strides @10-12
    bqEncoder.setBuffer(nil, offset: 0, index: 10)
    bqEncoder.setBuffer(nil, offset: 0, index: 11)
    bqEncoder.setBuffer(nil, offset: 0, index: 12)
    // multihead params @13-16
    var bqNumHeads = descriptor.queryShape.numHeads
    var bqNumKVHeads = descriptor.keyShape.numHeads
    var bqHeadDim = UInt32(descriptor.queryShape.headDimension)
    var bqSeqLen = descriptor.queryShape.sequenceLength
    bqEncoder.setBytes(&bqNumHeads, length: 4, index: 13)
    bqEncoder.setBytes(&bqNumKVHeads, length: 4, index: 14)
    bqEncoder.setBytes(&bqHeadDim, length: 4, index: 15)
    bqEncoder.setBytes(&bqSeqLen, length: 4, index: 16)
    bqEncoder.setBuffer(nil, offset: 0, index: 17)
    bqEncoder.setThreadgroupMemoryLength(Int(bqKernel.threadgroupMemoryAllocation), index: 0)

    let bqBlockCount = ceilDivide(
      Int(descriptor.queryShape.sequenceLength),
      Int(bqKernel.blockDimensions.parallelization)
    )
    let bqGrid = MTLSize(
      width: bqBlockCount,
      height: Int(descriptor.queryShape.numHeads),
      depth: Int(descriptor.queryShape.batchSize)
    )
    let bqGroup = MTLSize(width: Int(bqKernel.threadgroupSize), height: 1, depth: 1)
    bqEncoder.dispatchThreadgroups(bqGrid, threadsPerThreadgroup: bqGroup)
    bqEncoder.endEncoding()

    // --- Phase 2: backwardKeyValue (computes dK + dV, depends on D) ---
    // Buffer bindings:
    // Q@0 K@1 V@2 L@4 D@5 dO@6 dV@7 dK@8
    // nil strides @9 @10 @11
    // num_heads@12 num_kv_heads@13 head_dim@14 seq_len@15
    // nil mask@16
    let bkvKernelDescs = descriptor.kernelDescriptors(type: .backwardKeyValue)
    let bkvKernel = AttentionKernel(descriptor: bkvKernelDescs[0])

    guard
      let bkvPipeline = getOrCreatePipelineState(for: bkvKernel, descriptor: descriptor),
      let bkvEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      return nil
    }

    bkvEncoder.setComputePipelineState(bkvPipeline)
    bkvEncoder.setBuffer(query, offset: 0, index: 0)
    bkvEncoder.setBuffer(key, offset: 0, index: 1)
    bkvEncoder.setBuffer(value, offset: 0, index: 2)
    bkvEncoder.setBuffer(logsumexp, offset: 0, index: 4)
    bkvEncoder.setBuffer(dBuffer, offset: 0, index: 5)
    bkvEncoder.setBuffer(dOutput, offset: 0, index: 6)
    bkvEncoder.setBuffer(dValue, offset: 0, index: 7)
    bkvEncoder.setBuffer(dKey, offset: 0, index: 8)
    // nil strides @9-11
    bkvEncoder.setBuffer(nil, offset: 0, index: 9)
    bkvEncoder.setBuffer(nil, offset: 0, index: 10)
    bkvEncoder.setBuffer(nil, offset: 0, index: 11)
    // multihead params @12-15
    var bkvNumHeads = descriptor.queryShape.numHeads
    var bkvNumKVHeads = descriptor.keyShape.numHeads
    var bkvHeadDim = UInt32(descriptor.queryShape.headDimension)
    var bkvSeqLen = descriptor.queryShape.sequenceLength
    bkvEncoder.setBytes(&bkvNumHeads, length: 4, index: 12)
    bkvEncoder.setBytes(&bkvNumKVHeads, length: 4, index: 13)
    bkvEncoder.setBytes(&bkvHeadDim, length: 4, index: 14)
    bkvEncoder.setBytes(&bkvSeqLen, length: 4, index: 15)
    bkvEncoder.setBuffer(nil, offset: 0, index: 16)
    bkvEncoder.setThreadgroupMemoryLength(Int(bkvKernel.threadgroupMemoryAllocation), index: 0)

    let bkvBlockCount = ceilDivide(
      Int(descriptor.keyShape.sequenceLength),
      Int(bkvKernel.blockDimensions.parallelization)
    )
    let bkvGrid = MTLSize(
      width: bkvBlockCount,
      height: Int(descriptor.queryShape.numHeads),
      depth: Int(descriptor.queryShape.batchSize)
    )
    let bkvGroup = MTLSize(width: Int(bkvKernel.threadgroupSize), height: 1, depth: 1)
    bkvEncoder.dispatchThreadgroups(bkvGrid, threadsPerThreadgroup: bkvGroup)
    bkvEncoder.endEncoding()

    return commandBuffer
  }
}

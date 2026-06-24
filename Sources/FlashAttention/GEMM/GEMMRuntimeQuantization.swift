//
//  GEMMRuntimeQuantization.swift
//  FlashAttention
//
//  Swift interface for launching fused blockwise quantization kernels
//

import Metal

/// Runtime quantization utilities for efficient GPU-based quantization
public class GEMMRuntimeQuantization {
  private let device: MTLDevice
  private let library: MTLLibrary
  private var pipelineCache: [String: MTLComputePipelineState] = [:]

  public init(device: MTLDevice) throws {
    self.device = device
    guard let library = device.makeDefaultLibrary() else {
      throw QuantizationError.libraryNotFound
    }
    self.library = library
  }

  /// Errors that can occur during quantization
  public enum QuantizationError: Error {
    case libraryNotFound
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferCreationFailed
    case invalidParameters(String)
  }

  /// Get or create compute pipeline for the specified kernel
  private func getPipeline(for kernelName: String) throws -> MTLComputePipelineState {
    if let cached = pipelineCache[kernelName] {
      return cached
    }

    guard let function = library.makeFunction(name: kernelName) else {
      throw QuantizationError.functionNotFound(kernelName)
    }

    do {
      let pipeline = try device.makeComputePipelineState(function: function)
      pipelineCache[kernelName] = pipeline
      return pipeline
    } catch {
      throw QuantizationError.pipelineCreationFailed(kernelName)
    }
  }

  /// Perform fused blockwise centered quantization
  /// - Parameters:
  ///   - input: Input buffer containing floating point data
  ///   - inputPrecision: Precision of input data (FP32, FP16, or BF16)
  ///   - output: Output buffer for quantized INT8 data
  ///   - blockScales: Output buffer for per-block scales
  ///   - blockZeroPoints: Output buffer for per-block zero points
  ///   - precomputedSums: Optional output buffer for precomputed block sums
  ///   - K: Total number of elements
  ///   - blockSizeK: Size of each quantization block
  ///   - commandBuffer: Metal command buffer for execution
  public func quantizeBlockwiseCentered(
    input: MTLBuffer,
    inputPrecision: GEMMOperandPrecision,
    output: MTLBuffer,
    blockScales: MTLBuffer,
    blockZeroPoints: MTLBuffer,
    precomputedSums: MTLBuffer?,
    K: Int,
    blockSizeK: Int,
    commandBuffer: MTLCommandBuffer
  ) throws {
    // Validate parameters
    guard blockSizeK > 0, blockSizeK % 8 == 0 else {
      throw QuantizationError
        .invalidParameters("blockSizeK must be positive and multiple of 8")
    }

    guard K > 0 else {
      throw QuantizationError.invalidParameters("K must be positive")
    }

    // Select appropriate kernel based on input precision
    let kernelName: String
    switch inputPrecision {
    case .FP32:
      kernelName = "quantize_blockwise_centered_fp32_to_int8"
    case .FP16:
      kernelName = "quantize_blockwise_centered_fp16_to_int8"
    case .BF16:
      kernelName = "quantize_blockwise_centered_bf16_to_int8"
    default:
      throw QuantizationError
        .invalidParameters("Unsupported input precision: \(inputPrecision)")
    }

    let pipeline = try getPipeline(for: kernelName)

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw QuantizationError.pipelineCreationFailed("Failed to create compute encoder")
    }

    encoder.setComputePipelineState(pipeline)

    // Set buffers
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBuffer(blockScales, offset: 0, index: 2)
    encoder.setBuffer(blockZeroPoints, offset: 0, index: 3)

    if let precomputedSums {
      encoder.setBuffer(precomputedSums, offset: 0, index: 4)
    } else {
      // Pass null pointer for optional buffer
      encoder.setBuffer(input, offset: 0, index: 4) // Dummy buffer
    }

    // Set constants
    var K_uint = UInt32(K)
    var blockSizeK_uint = UInt32(blockSizeK)
    encoder.setBytes(&K_uint, length: MemoryLayout<UInt32>.size, index: 5)
    encoder.setBytes(&blockSizeK_uint, length: MemoryLayout<UInt32>.size, index: 6)

    // Calculate dispatch parameters
    let numBlocks = (K + blockSizeK - 1) / blockSizeK
    let threadsPerBlock = min(256, blockSizeK) // Use up to 256 threads per block
    let totalThreads = numBlocks * threadsPerBlock

    let threadsPerThreadgroup = MTLSize(width: threadsPerBlock, height: 1, depth: 1)
    let threadgroupsPerGrid = MTLSize(
      width: (totalThreads + threadsPerBlock - 1) / threadsPerBlock,
      height: 1,
      depth: 1
    )

    encoder.dispatchThreadgroups(
      threadgroupsPerGrid,
      threadsPerThreadgroup: threadsPerThreadgroup
    )
    encoder.endEncoding()
  }

  /// Create buffers needed for blockwise quantization
  /// - Parameters:
  ///   - elementCount: Number of elements to quantize
  ///   - blockSizeK: Size of quantization blocks
  ///   - includePrecomputedSums: Whether to create buffer for precomputed sums
  /// - Returns: Tuple of (scales, zeroPoints, optionalSums) buffers
  public func createBlockwiseBuffers(
    elementCount: Int,
    blockSizeK: Int,
    includePrecomputedSums: Bool = false
  ) throws
    -> (scales: MTLBuffer, zeroPoints: MTLBuffer, sums: MTLBuffer?)
  {
    let numBlocks = (elementCount + blockSizeK - 1) / blockSizeK

    guard
      let scalesBuffer = device.makeBuffer(
        length: numBlocks * MemoryLayout<Float>.size,
        options: .storageModeShared
      )
    else {
      throw QuantizationError.bufferCreationFailed
    }

    guard
      let zeroPointsBuffer = device.makeBuffer(
        length: numBlocks * MemoryLayout<Int8>.size,
        options: .storageModeShared
      )
    else {
      throw QuantizationError.bufferCreationFailed
    }

    let sumsBuffer: MTLBuffer?
    if includePrecomputedSums {
      sumsBuffer = device.makeBuffer(
        length: numBlocks * MemoryLayout<Int32>.size,
        options: .storageModeShared
      )
      guard sumsBuffer != nil else {
        throw QuantizationError.bufferCreationFailed
      }
    } else {
      sumsBuffer = nil
    }

    return (scales: scalesBuffer, zeroPoints: zeroPointsBuffer, sums: sumsBuffer)
  }

  /// Extract quantization parameters from blockwise buffers
  /// - Parameters:
  ///   - scalesBuffer: Buffer containing per-block scales
  ///   - zeroPointsBuffer: Buffer containing per-block zero points
  ///   - numBlocks: Number of blocks
  /// - Returns: Arrays of scales and zero points
  public func extractBlockwiseParameters(
    scalesBuffer: MTLBuffer,
    zeroPointsBuffer: MTLBuffer,
    numBlocks: Int
  )
    -> (scales: [Float], zeroPoints: [Int32])
  {
    let scalesPtr = scalesBuffer.contents().bindMemory(to: Float.self, capacity: numBlocks)
    let zeroPointsPtr = zeroPointsBuffer.contents().bindMemory(
      to: Int8.self,
      capacity: numBlocks
    )

    let scales = Array(UnsafeBufferPointer(start: scalesPtr, count: numBlocks))
    let zeroPoints = Array(UnsafeBufferPointer(start: zeroPointsPtr, count: numBlocks))
      .map { Int32($0) }

    return (scales: scales, zeroPoints: zeroPoints)
  }

  /// Utility method to quantize a tensor using fused blockwise centered quantization
  /// - Parameters:
  ///   - inputBuffer: Input floating point buffer
  ///   - inputPrecision: Precision of input data
  ///   - elementCount: Number of elements
  ///   - blockSizeK: Size of quantization blocks
  ///   - commandBuffer: Command buffer for execution
  /// - Returns: QuantizedTensor with blockwise quantization parameters
  public func quantizeBlockwiseCenteredTensor(
    inputBuffer: MTLBuffer,
    inputPrecision: GEMMOperandPrecision,
    elementCount: Int,
    blockSizeK: Int,
    commandBuffer: MTLCommandBuffer
  ) throws
    -> QuantizedTensor
  {
    // Create output buffers
    guard
      let quantizedBuffer = device.makeBuffer(
        length: elementCount,
        options: .storageModeShared
      )
    else {
      throw QuantizationError.bufferCreationFailed
    }

    let (scalesBuffer, zeroPointsBuffer, _) = try createBlockwiseBuffers(
      elementCount: elementCount,
      blockSizeK: blockSizeK,
      includePrecomputedSums: false
    )

    // Perform quantization
    try quantizeBlockwiseCentered(
      input: inputBuffer,
      inputPrecision: inputPrecision,
      output: quantizedBuffer,
      blockScales: scalesBuffer,
      blockZeroPoints: zeroPointsBuffer,
      precomputedSums: nil,
      K: elementCount,
      blockSizeK: blockSizeK,
      commandBuffer: commandBuffer
    )

    // Wait for completion to read parameters
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Extract parameters
    let numBlocks = (elementCount + blockSizeK - 1) / blockSizeK
    let (scales, zeroPoints) = extractBlockwiseParameters(
      scalesBuffer: scalesBuffer,
      zeroPointsBuffer: zeroPointsBuffer,
      numBlocks: numBlocks
    )

    // Create quantization parameters
    let quantParams = QuantizationParameters(
      scales: scales,
      zeroPoints: zeroPoints,
      precision: .INT8,
      mode: .blockwise(blockSizeK: blockSizeK, bothOperands: false),
      strategy: .symmetric
    )

    return QuantizedTensor(
      device: device,
      data: quantizedBuffer,
      parameters: quantParams,
      elementCount: elementCount,
      shape: [elementCount], // Assuming 1D for simplicity
      blockScales: scalesBuffer,
      blockZeroPoints: zeroPointsBuffer,
      blockSizeK: blockSizeK,
      precomputedSums: nil
    )
  }
}

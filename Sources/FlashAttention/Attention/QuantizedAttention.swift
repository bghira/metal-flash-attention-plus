//
//  QuantizedAttention.swift
//  FlashAttention
//
//

import Metal

/// Quantized Flash Attention implementation with GPU acceleration
public class QuantizedAttention {
  /// Quantized attention configuration
  public struct Configuration {
    /// Precision for Query tensor
    public var queryPrecision: GEMMOperandPrecision = .FP16

    /// Precision for Key tensor
    public var keyPrecision: GEMMOperandPrecision = .INT8

    /// Precision for Value tensor
    public var valuePrecision: GEMMOperandPrecision = .INT8

    /// Quantization strategy for Query tensor
    public var queryStrategy: QuantizationStrategy = .legacy

    /// Quantization strategy for Key tensor
    public var keyStrategy: QuantizationStrategy = .legacy

    /// Quantization strategy for Value tensor
    public var valueStrategy: QuantizationStrategy = .legacy

    /// Serialized strategy version for forward compatibility
    public var strategyVersion: UInt8 = QuantizationStrategy.currentVersion

    /// Whether to use mixed precision intermediate computations
    public var mixedPrecisionIntermediates: Bool = true

    /// Quantization parameters for each tensor
    public var quantizationParameters: [String: QuantizationParameters] = [:]

    public init() {}
  }

  public struct QuantizedBufferManifest {
    public struct ActiveRange {
      public var start: UInt32
      public var end: UInt32
    }

    public private(set) var activeRanges: [ActiveRange] = []

    public mutating func addActiveRange(_ range: Range<UInt32>) {
      guard !range.isEmpty else { return }
      activeRanges.append(ActiveRange(start: range.lowerBound, end: range.upperBound))
    }
  }

  /// Quantized attention descriptor that extends AttentionDescriptor
  public struct QuantizedAttentionDescriptor {
    /// Base attention descriptor
    public var baseDescriptor: AttentionDescriptor

    /// Quantization configuration
    public var quantizationConfig: Configuration

    public init(baseDescriptor: AttentionDescriptor, quantizationConfig: Configuration) {
      self.baseDescriptor = baseDescriptor
      self.quantizationConfig = quantizationConfig
    }

    /// Generate kernel descriptor with quantized precision handling
    public func kernelDescriptor(type: AttentionKernelType) -> AttentionKernelDescriptor {
      var descriptor = baseDescriptor.kernelDescriptor(type: type)

      // Override memory precisions with quantized settings
      descriptor.memoryPrecisions[.Q] = quantizationConfig.queryPrecision
      descriptor.memoryPrecisions[.K] = quantizationConfig.keyPrecision
      descriptor.memoryPrecisions[.V] = quantizationConfig.valuePrecision

      // Set register precisions to FP32 for quantized inputs
      if quantizationConfig.queryPrecision.requiresQuantizationParameters {
        descriptor.registerPrecisions[.Q] = .FP32
      }
      if quantizationConfig.keyPrecision.requiresQuantizationParameters {
        descriptor.registerPrecisions[.K] = .FP32
      }
      if quantizationConfig.valuePrecision.requiresQuantizationParameters {
        descriptor.registerPrecisions[.V] = .FP32
      }

      return descriptor
    }
  }

  private let device: MTLDevice
  private let commandQueue: MTLCommandQueue?
  private var pipelineCache: [String: MTLComputePipelineState] = [:]
  private var isDisposed: Bool = false

  /// Set to true via the MFA_DEBUG environment variable to enable verbose
  /// diagnostic prints (quantization parameters, buffer sizes, samples).
  private static let debugEnabled: Bool = {
    ProcessInfo.processInfo.environment["MFA_DEBUG"] != nil
  }()

  public init(device: MTLDevice) {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      fatalError("Could not create Metal command queue")
    }
    commandQueue = queue
  }

  /// Safe disposal method to prevent crashes during Swift ARC cleanup
  private func dispose() {
    guard !isDisposed else { return }

    // Clear pipeline cache safely
    pipelineCache.removeAll()

    // Mark as disposed to prevent double-cleanup
    isDisposed = true
  }

  /// Swift deinitializer with defensive guards
  deinit {
    dispose()
  }

  /// Perform quantized attention forward pass
  /// - Parameters:
  ///   - query: Query tensor (can be FP32, FP16, or quantized)
  ///   - key: Key tensor (can be FP32, FP16, or quantized)
  ///   - value: Value tensor (can be FP32, FP16, or quantized)
  ///   - output: Output tensor buffer
  ///   - descriptor: Quantized attention configuration
  ///   - bufferOffsets: Byte offsets into Q/K/V/O/LSE buffers (for multi-head slicing)
  ///   - externalLogsumexp: Optional externally-provided LSE buffer
  /// - Returns: Command buffer for execution
  public func forward(
    query: QuantizedTensor,
    key: QuantizedTensor,
    value: QuantizedTensor,
    output: MTLBuffer,
    descriptor: QuantizedAttentionDescriptor,
    bufferOffsets: (q: Int, k: Int, v: Int, o: Int, lse: Int) = (0, 0, 0, 0, 0),
    externalLogsumexp: MTLBuffer? = nil
  )
    -> MTLCommandBuffer?
  {
    guard
      !isDisposed, let queue = commandQueue,
      let commandBuffer = queue.makeCommandBuffer()
    else {
      print("Error: Failed to create command buffer (disposed: \(isDisposed))")
      return nil
    }

    let kernelDescriptor = descriptor.kernelDescriptor(type: AttentionKernelType.forward)

    let kernel = AttentionKernel(descriptor: kernelDescriptor)

    if
      let sparseMask = descriptor.baseDescriptor.sparseMask,
      case .sparseRanges = sparseMask.maskType,
      let kvRanges = sparseMask.maskBuffer,
      let matrixDims = descriptor.baseDescriptor.matrixDimensions
    {
      let manifest = prepareSparseQuantizedBuffers(
        kvRanges: kvRanges,
        quantizationParams: key.parameters,
        sequenceLength: Int(matrixDims.column)
      )
      _ = manifest
    }

    // Create pipeline state for quantized attention
    guard
      let pipelineState = getOrCreatePipelineState(
        for: kernel,
        descriptor: descriptor,
        operands: (query, key, value)
      )
    else {
      print("Error: Failed to create pipeline state")
      return nil
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    encoder.setComputePipelineState(pipelineState)

    // Set threadgroup memory length (required for flash attention kernels)
    let threadgroupMemoryLength = Int(kernel.threadgroupMemoryAllocation)
    encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)

    // Bind buffers in the exact order produced by createBufferBindings():
    //   Q@0 K@1 V@2 O@3 L@4
    //   per quantized operand (Q,K,V order): scale, zero_point, strategy, strategy_version
    //   per quantized operand: block_scales, block_zero_points
    //   q/k/v/o strides, num_heads/num_kv_heads/head_dim/seq_len, mask  (left null → single-head)
    // The previous dispatch started quant params at index 4 (where L belongs),
    // set M/N/K at positions the kernel doesn't use, and dispatched a single
    // threadgroup regardless of sequence length.
    encoder.setBuffer(query.data, offset: bufferOffsets.q, index: 0)
    encoder.setBuffer(key.data, offset: bufferOffsets.k, index: 1)
    encoder.setBuffer(value.data, offset: bufferOffsets.v, index: 2)
    encoder.setBuffer(output, offset: bufferOffsets.o, index: 3)

    let dims = descriptor.baseDescriptor.matrixDimensions!
    let sequenceLength = UInt32(dims.row)

    let logsumexpBuffer = externalLogsumexp ?? device.makeBuffer(
      length: Int(sequenceLength) * MemoryLayout<Float>.size,
      options: .storageModePrivate
    )
    encoder.setBuffer(logsumexpBuffer, offset: bufferOffsets.lse, index: 4)

    // Quantized operands in Q, K, V order (matches createBufferBindings' sort).
    let quantOperands: [QuantizedTensor] = [query, key, value]
      .filter(\.parameters.precision.requiresQuantizationParameters)

    var bufferIndex = 5
    for operand in quantOperands {
      var scale = operand.parameters.scale
      var zeroPoint = Int32(operand.parameters.zeroPoint)
      encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: bufferIndex)
      bufferIndex += 1
      encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: bufferIndex)
      bufferIndex += 1
    }
    for operand in quantOperands {
      encoder.setBuffer(operand.blockScales, offset: 0, index: bufferIndex)
      bufferIndex += 1
      encoder.setBuffer(operand.blockZeroPoints, offset: 0, index: bufferIndex)
      bufferIndex += 1
    }

    // Use proper threadgroup configuration from AttentionKernel
    let kernelThreadgroupSize = Int(kernel.threadgroupSize)
    let blockParallelization = Int(kernel.blockDimensions.parallelization)

    // Grid covers the full row (query) dimension: one threadgroup per
    // parallelization tile.
    let blockCount = (Int(sequenceLength) + blockParallelization - 1) / blockParallelization
    let threadgroupSize = MTLSize(width: kernelThreadgroupSize, height: 1, depth: 1)
    let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    return commandBuffer
  }

  /// Perform quantized attention forward pass with runtime quantization
  /// - Parameters:
  ///   - queryBuffer: Query tensor buffer containing fp16/bf16/fp32 data
  ///   - keyBuffer: Key tensor buffer containing fp16/bf16/fp32 data
  ///   - valueBuffer: Value tensor buffer containing fp16/bf16/fp32 data
  ///   - output: Output tensor buffer
  ///   - queryPrecision: Input precision of query buffer (FP16, BF16, FP32)
  ///   - keyPrecision: Input precision of key buffer (FP16, BF16, FP32)
  ///   - valuePrecision: Input precision of value buffer (FP16, BF16, FP32)
  ///   - targetQuantization: Target quantization precision (INT8, INT4)
  ///   - quantizationMode: Quantization granularity mode
  ///   - descriptor: Quantized attention configuration
  /// - Returns: Command buffer for execution
  public func forward(
    queryBuffer: MTLBuffer,
    keyBuffer: MTLBuffer,
    valueBuffer: MTLBuffer,
    output: MTLBuffer,
    queryShape: [Int],
    keyShape: [Int],
    valueShape: [Int],
    queryPrecision: GEMMOperandPrecision,
    keyPrecision: GEMMOperandPrecision,
    valuePrecision: GEMMOperandPrecision,
    targetQuantization: GEMMOperandPrecision,
    quantizationMode: QuantizationMode = .tensorWise,
    descriptor: QuantizedAttentionDescriptor
  )
    -> MTLCommandBuffer?
  {
    guard !isDisposed, let _ = commandQueue else {
      print("Error: Failed to access command queue (disposed: \(isDisposed))")
      return nil
    }

    // Convert input buffers to quantized tensors at runtime
    let quantizedQuery = createQuantizedTensorFromBuffer(
      buffer: queryBuffer,
      shape: queryShape,
      inputPrecision: queryPrecision,
      targetPrecision: targetQuantization,
      quantizationMode: quantizationMode,
      targetStrategy: descriptor.quantizationConfig.queryStrategy
    )

    let quantizedKey = createQuantizedTensorFromBuffer(
      buffer: keyBuffer,
      shape: keyShape,
      inputPrecision: keyPrecision,
      targetPrecision: targetQuantization,
      quantizationMode: quantizationMode,
      targetStrategy: descriptor.quantizationConfig.keyStrategy
    )

    let quantizedValue = createQuantizedTensorFromBuffer(
      buffer: valueBuffer,
      shape: valueShape,
      inputPrecision: valuePrecision,
      targetPrecision: targetQuantization,
      quantizationMode: quantizationMode,
      targetStrategy: descriptor.quantizationConfig.valueStrategy
    )

    // Use existing forward method with quantized tensors
    return forward(
      query: quantizedQuery,
      key: quantizedKey,
      value: quantizedValue,
      output: output,
      descriptor: descriptor
    )
  }

  /// Simplified overload with uniform quantization settings
  /// - Parameters:
  ///   - queryBuffer: Query tensor buffer containing fp16/bf16/fp32 data
  ///   - keyBuffer: Key tensor buffer containing fp16/bf16/fp32 data
  ///   - valueBuffer: Value tensor buffer containing fp16/bf16/fp32 data
  ///   - output: Output tensor buffer
  ///   - inputPrecision: Common input precision for all tensors (FP16, BF16, FP32)
  ///   - targetQuantization: Target quantization precision (INT8, INT4)
  ///   - quantizationMode: Quantization granularity mode
  ///   - descriptor: Quantized attention configuration
  /// - Returns: Command buffer for execution
  public func forward(
    queryBuffer: MTLBuffer,
    keyBuffer: MTLBuffer,
    valueBuffer: MTLBuffer,
    output: MTLBuffer,
    tensorShape: [Int],
    inputPrecision: GEMMOperandPrecision,
    targetQuantization: GEMMOperandPrecision,
    quantizationMode: QuantizationMode = .tensorWise,
    descriptor: QuantizedAttentionDescriptor
  )
    -> MTLCommandBuffer?
  {
    forward(
      queryBuffer: queryBuffer,
      keyBuffer: keyBuffer,
      valueBuffer: valueBuffer,
      output: output,
      queryShape: tensorShape,
      keyShape: tensorShape,
      valueShape: tensorShape,
      queryPrecision: inputPrecision,
      keyPrecision: inputPrecision,
      valuePrecision: inputPrecision,
      targetQuantization: targetQuantization,
      quantizationMode: quantizationMode,
      descriptor: descriptor
    )
  }

  /// Public wrapper for runtime quantization from an FP32/FP16/BF16 buffer.
  /// Used by the multi-head quantized FFI to quantize full BHSD tensors.
  public func createQuantizedTensorFromBufferPublic(
    buffer: MTLBuffer,
    shape: [Int],
    inputPrecision: GEMMOperandPrecision,
    targetPrecision: GEMMOperandPrecision,
    quantizationMode: QuantizationMode,
    targetStrategy: QuantizationStrategy
  )
    -> QuantizedTensor?
  {
    guard targetPrecision.requiresQuantizationParameters else {
      let parameters = QuantizationParameters(
        scale: 1.0, zeroPoint: 0,
        precision: targetPrecision,
        mode: quantizationMode,
        strategy: targetStrategy
      )
      return QuantizedTensor(
        device: device, data: buffer, parameters: parameters,
        elementCount: shape.reduce(1, *), shape: shape
      )
    }
    return createQuantizedTensorFromBuffer(
      buffer: buffer, shape: shape,
      inputPrecision: inputPrecision,
      targetPrecision: targetPrecision,
      quantizationMode: quantizationMode,
      targetStrategy: targetStrategy
    )
  }

  /// Helper method to create quantized tensor from existing buffer with runtime quantization
  private func createQuantizedTensorFromBuffer(
    buffer: MTLBuffer,
    shape: [Int],
    inputPrecision: GEMMOperandPrecision,
    targetPrecision: GEMMOperandPrecision,
    quantizationMode: QuantizationMode,
    targetStrategy: QuantizationStrategy
  )
    -> QuantizedTensor
  {
    let elementCount = shape.reduce(1, *)

    // If target precision doesn't require quantization, wrap existing buffer
    guard targetPrecision.requiresQuantizationParameters else {
      let parameters = QuantizationParameters(
        scale: 1.0,
        zeroPoint: 0,
        precision: targetPrecision,
        mode: quantizationMode,
        strategy: targetStrategy
      )
      return QuantizedTensor(
        device: device,
        data: buffer,
        parameters: parameters,
        elementCount: elementCount,
        shape: shape
      )
    }

    // Use fused quantization for symmetric blockwise quantization
    if
      targetStrategy == .symmetric,
      case let .blockwise(blockSizeK, _) = quantizationMode,
      targetPrecision == .INT8
    {
      do {
        // Initialize the runtime quantization utility
        let runtimeQuantizer = try GEMMRuntimeQuantization(device: device)

        // Create command buffer for fused quantization
        guard
          let commandQueue = device.makeCommandQueue(),
          let commandBuffer = commandQueue.makeCommandBuffer()
        else {
          fatalError("Could not create Metal command queue or command buffer")
        }

        // Use fused blockwise centered quantization
        let quantizedTensor = try runtimeQuantizer.quantizeBlockwiseCenteredTensor(
          inputBuffer: buffer,
          inputPrecision: inputPrecision,
          elementCount: elementCount,
          blockSizeK: blockSizeK,
          commandBuffer: commandBuffer
        )

        return quantizedTensor
      } catch {
        if Self.debugEnabled {
          print("Warning: Fused quantization failed, falling back to CPU quantization: \(error)")
        }
        // Fall through to CPU quantization below
      }
    }

    // Fallback to CPU-based quantization for other strategies
    // Convert input buffer to Float array for quantization parameter calculation
    let floatData = convertBufferToFloat(
      buffer: buffer,
      elementCount: elementCount,
      inputPrecision: inputPrecision
    )

    // Calculate quantization parameters based on mode
    let parameters = floatData.withUnsafeBufferPointer { floatPtr in
      guard let baseAddress = floatPtr.baseAddress else {
        fatalError("Failed to obtain base address from converted float data")
      }
      return targetPrecision.calculateQuantizationParameters(
        data: baseAddress,
        count: elementCount,
        shape: shape,
        mode: quantizationMode,
        strategy: targetStrategy
      )
    }

    // Create quantized buffer
    let bufferSize = targetPrecision == .INT4 ? (elementCount + 1) / 2 : elementCount *
      targetPrecision.size
    guard let quantizedBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
    else {
      fatalError("Could not create quantized buffer")
    }

    // Quantize the data
    floatData.withUnsafeBufferPointer { floatPtr in
      targetPrecision.quantize(
        input: floatPtr.baseAddress!,
        output: quantizedBuffer.contents(),
        count: elementCount,
        parameters: parameters
      )
    }

    return QuantizedTensor(
      device: device,
      data: quantizedBuffer,
      parameters: parameters,
      elementCount: elementCount,
      shape: shape
    )
  }

  /// Convert Metal buffer data to Float array based on input precision
  private func convertBufferToFloat(
    buffer: MTLBuffer,
    elementCount: Int,
    inputPrecision: GEMMOperandPrecision
  )
    -> [Float]
  {
    var floatData = [Float](repeating: 0, count: elementCount)
    let bufferContents = buffer.contents()

    if Self.debugEnabled {
      print("🔍 convertBufferToFloat: precision=\(inputPrecision), elementCount=\(elementCount)")
      print("🔍 buffer.length=\(buffer.length), expected=\(elementCount * inputPrecision.size)")
    }

    // Validate buffer size
    let expectedSize = elementCount * inputPrecision.size
    guard buffer.length >= expectedSize else {
      print("❌ Buffer size mismatch: got \(buffer.length), expected \(expectedSize)")
      return floatData // Return zeros on error
    }

    switch inputPrecision {
    case .FP32:
      let floatPtr = bufferContents.bindMemory(to: Float.self, capacity: elementCount)
      for i in 0..<elementCount {
        floatData[i] = floatPtr[i]
      }

    case .FP16:
      let halfPtr = bufferContents.bindMemory(to: Float16.self, capacity: elementCount)
      for i in 0..<elementCount {
        floatData[i] = Float(halfPtr[i])
      }

    case .BF16:
      // PyTorch stores BF16 as uint16 values in memory
      let bfloat16Ptr = bufferContents.bindMemory(to: UInt16.self, capacity: elementCount)
      for i in 0..<elementCount {
        // Convert BF16 to FP32 by shifting left 16 bits and padding with zeros
        let bfloat16Value = bfloat16Ptr[i]
        let fp32Bits = UInt32(bfloat16Value) << 16
        floatData[i] = Float(bitPattern: fp32Bits)
      }

    default:
      print("❌ Unsupported input precision for runtime quantization: \(inputPrecision)")
      fatalError("Unsupported input precision for runtime quantization: \(inputPrecision)")
    }

    if Self.debugEnabled {
      let sampleCount = min(4, elementCount)
      let sampleValues = Array(floatData.prefix(sampleCount))
      print("🔍 Converted first \(sampleCount) values: \(sampleValues)")
    }

    return floatData
  }

  private func getOrCreatePipelineState(
    for kernel: AttentionKernel,
    descriptor: QuantizedAttentionDescriptor,
    operands: (query: QuantizedTensor, key: QuantizedTensor, value: QuantizedTensor)
  )
    -> MTLComputePipelineState?
  {
    let source = kernel.createSource()
    func isBlockwise(_ tensor: QuantizedTensor) -> Bool {
      if case .blockwise = tensor.parameters.mode {
        return true
      }
      return false
    }

    let queryBlockwise = isBlockwise(operands.query) && operands.query.blockScales != nil
    if queryBlockwise {
      print(
        "Warning: Blockwise quantization for query tensors is not yet supported; falling back to per-tensor scaling."
      )
    }
    var hasBlockwiseQ = false
    var hasBlockwiseK = isBlockwise(operands.key) && operands.key.blockScales != nil
    var hasBlockwiseV = isBlockwise(operands.value) && operands.value.blockScales != nil

    var blockSize = operands.key.blockSizeK ?? operands.value.blockSizeK ?? operands.query
      .blockSizeK ?? 0
    if blockSize == 0 {
      blockSize = 1
    }
    var blockSizeUInt = UInt32(blockSize)

    let cacheKey =
      "\(source.hashValue)_\(hasBlockwiseQ ? 1 : 0)_\(hasBlockwiseK ? 1 : 0)_\(hasBlockwiseV ? 1 : 0)_\(blockSizeUInt)"

    if let cached = pipelineCache[cacheKey] {
      return cached
    }

    do {
      let library = try device.makeLibrary(source: source, options: nil)

      let functionConstants = MTLFunctionConstantValues()
      descriptor.baseDescriptor.setFunctionConstants(functionConstants)
      functionConstants.setConstantValue(&hasBlockwiseQ, type: .bool, index: 5)
      functionConstants.setConstantValue(&hasBlockwiseK, type: .bool, index: 6)
      functionConstants.setConstantValue(&hasBlockwiseV, type: .bool, index: 7)
      functionConstants.setConstantValue(&blockSizeUInt, type: .uint, index: 8)

      // DEBUG: Check function constants after setting

      let function = try library.makeFunction(name: "attention", constantValues: functionConstants)
      let pipelineState = try device.makeComputePipelineState(function: function)

      pipelineCache[cacheKey] = pipelineState
      return pipelineState
    } catch {
      print("Pipeline creation error: \(error)")
      return nil
    }
  }
}

public extension QuantizedAttention {
  func prepareSparseQuantizedBuffers(
    kvRanges: MTLBuffer,
    quantizationParams: QuantizationParameters,
    sequenceLength: Int
  )
    -> QuantizedBufferManifest
  {
    var manifest = QuantizedBufferManifest()

    let elementStride = MemoryLayout<UInt32>.stride * 2
    guard kvRanges.length % elementStride == 0 else {
      return manifest
    }

    let entryCount = kvRanges.length / elementStride
    guard entryCount > 0 else {
      return manifest
    }

    kvRanges.contents().withMemoryRebound(to: UInt32.self, capacity: entryCount * 2) { pointer in
      for idx in 0..<entryCount {
        let lower = pointer[idx * 2]
        let upper = pointer[idx * 2 + 1]

        let clampedStart = min(lower, UInt32(sequenceLength))
        let clampedEnd = min(max(upper, clampedStart), UInt32(sequenceLength))

        if clampedStart < clampedEnd {
          manifest.addActiveRange(clampedStart..<clampedEnd)
        }
      }
    }

    _ = quantizationParams // Placeholder for future selective dequant logic
    return manifest
  }
}

extension QuantizedAttention.Configuration: Codable {
  private enum CodingKeys: String, CodingKey {
    case queryPrecision
    case keyPrecision
    case valuePrecision
    case queryStrategy
    case keyStrategy
    case valueStrategy
    case strategyVersion
    case mixedPrecisionIntermediates
    case quantizationParameters
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    queryPrecision = try container.decodeIfPresent(
      GEMMOperandPrecision.self,
      forKey: .queryPrecision
    ) ?? .FP16
    keyPrecision = try container
      .decodeIfPresent(GEMMOperandPrecision.self, forKey: .keyPrecision) ?? .INT8
    valuePrecision = try container.decodeIfPresent(
      GEMMOperandPrecision.self,
      forKey: .valuePrecision
    ) ?? .INT8

    queryStrategy = try container.decodeIfPresent(
      QuantizationStrategy.self,
      forKey: .queryStrategy
    ) ?? .legacy
    keyStrategy = try container
      .decodeIfPresent(QuantizationStrategy.self, forKey: .keyStrategy) ?? .legacy
    valueStrategy = try container.decodeIfPresent(
      QuantizationStrategy.self,
      forKey: .valueStrategy
    ) ?? .legacy
    strategyVersion = try container
      .decodeIfPresent(UInt8.self, forKey: .strategyVersion) ?? QuantizationStrategy.currentVersion

    mixedPrecisionIntermediates = try container.decodeIfPresent(
      Bool.self,
      forKey: .mixedPrecisionIntermediates
    ) ?? true
    quantizationParameters = try container.decodeIfPresent(
      [String: QuantizationParameters].self,
      forKey: .quantizationParameters
    ) ?? [:]
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(queryPrecision, forKey: .queryPrecision)
    try container.encode(keyPrecision, forKey: .keyPrecision)
    try container.encode(valuePrecision, forKey: .valuePrecision)
    try container.encode(queryStrategy, forKey: .queryStrategy)
    try container.encode(keyStrategy, forKey: .keyStrategy)
    try container.encode(valueStrategy, forKey: .valueStrategy)
    try container.encode(strategyVersion, forKey: .strategyVersion)
    try container.encode(mixedPrecisionIntermediates, forKey: .mixedPrecisionIntermediates)
    try container.encode(quantizationParameters, forKey: .quantizationParameters)
  }
}

// MARK: - Convenience extensions

public extension QuantizedAttention {
  /// Ultra-simplified API for runtime quantization
  /// - Parameters:
  ///   - queryBuffer: Query tensor buffer (any supported floating-point format)
  ///   - keyBuffer: Key tensor buffer (any supported floating-point format)
  ///   - valueBuffer: Value tensor buffer (any supported floating-point format)
  ///   - output: Output tensor buffer
  ///   - shape: Common tensor shape [batch, sequence, head_dim]
  ///   - inputFormat: Input data format (FP16, BF16, or FP32)
  ///   - quantizeTo: Target quantization (INT8 or INT4)
  ///   - mode: Quantization granularity mode
  /// - Returns: Command buffer for execution
  func forwardWithRuntimeQuantization(
    queryBuffer: MTLBuffer,
    keyBuffer: MTLBuffer,
    valueBuffer: MTLBuffer,
    output: MTLBuffer,
    shape: [Int],
    inputFormat: GEMMOperandPrecision = .FP16,
    quantizeTo: GEMMOperandPrecision = .INT8,
    mode: QuantizationMode = .tensorWise
  )
    -> MTLCommandBuffer?
  {
    // Create default attention descriptor
    var baseDescriptor = AttentionDescriptor()
    guard shape.count >= 3 else {
      print("Error: Shape must have at least 3 dimensions [batch, sequence, head_dim]")
      return nil
    }

    let sequenceLength = shape[1]
    let headDim = shape[2]

    // Validate dimensions are positive
    guard sequenceLength > 0, headDim > 0 else {
      print("Error: Invalid dimensions - sequence: \(sequenceLength), headDim: \(headDim)")
      return nil
    }

    baseDescriptor.matrixDimensions = (
      row: UInt32(sequenceLength),
      column: UInt32(sequenceLength),
      head: UInt16(headDim)
    )
    baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

    // Create quantization configuration
    var config = Configuration()
    config.queryPrecision = quantizeTo
    config.keyPrecision = quantizeTo
    config.valuePrecision = quantizeTo

    let descriptor = QuantizedAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      quantizationConfig: config
    )

    return forward(
      queryBuffer: queryBuffer,
      keyBuffer: keyBuffer,
      valueBuffer: valueBuffer,
      output: output,
      tensorShape: shape,
      inputPrecision: inputFormat,
      targetQuantization: quantizeTo,
      quantizationMode: mode,
      descriptor: descriptor
    )
  }

  /// Create quantized tensors from floating point arrays
  /// - Parameters:
  ///   - queryData: Query data as Float array
  ///   - keyData: Key data as Float array
  ///   - valueData: Value data as Float array
  ///   - queryShape: Shape of query tensor
  ///   - keyShape: Shape of key tensor
  ///   - valueShape: Shape of value tensor
  ///   - config: Quantization configuration
  /// - Returns: Tuple of quantized tensors
  func createQuantizedTensors(
    queryData: [Float], keyData: [Float], valueData: [Float],
    queryShape: [Int], keyShape: [Int], valueShape: [Int],
    config: Configuration
  )
    -> (query: QuantizedTensor, key: QuantizedTensor, value: QuantizedTensor)
  {
    let query = QuantizedTensor.from(
      device: device,
      floatData: queryData,
      shape: queryShape,
      precision: config.queryPrecision,
      strategy: config.queryStrategy
    )

    let key = QuantizedTensor.from(
      device: device,
      floatData: keyData,
      shape: keyShape,
      precision: config.keyPrecision,
      strategy: config.keyStrategy
    )

    let value = QuantizedTensor.from(
      device: device,
      floatData: valueData,
      shape: valueShape,
      precision: config.valuePrecision,
      strategy: config.valueStrategy
    )

    return (query, key, value)
  }

  /// Benchmark quantized vs non-quantized attention
  /// - Parameters:
  ///   - batchSize: Batch size
  ///   - sequenceLength: Sequence length
  ///   - headDim: Head dimension
  ///   - iterations: Number of benchmark iterations
  /// - Returns: Dictionary with benchmark results
  func benchmark(
    batchSize: Int = 1,
    sequenceLength: Int = 1024,
    headDim: Int = 64,
    iterations: Int = 100
  )
    -> [String: Double]
  {
    let totalElements = batchSize * sequenceLength * headDim

    // Generate random test data
    let queryData = (0..<totalElements).map { _ in Float.random(in: -1...1) }
    let keyData = (0..<totalElements).map { _ in Float.random(in: -1...1) }
    let valueData = (0..<totalElements).map { _ in Float.random(in: -1...1) }

    let shape = [batchSize, sequenceLength, headDim]

    // Test configurations
    let configs: [String: Configuration] = [
      "FP16": {
        var config = Configuration()
        config.queryPrecision = .FP16
        config.keyPrecision = .FP16
        config.valuePrecision = .FP16
        return config
      }(),
      "INT8": {
        var config = Configuration()
        config.queryPrecision = .FP16
        config.keyPrecision = .INT8
        config.valuePrecision = .INT8
        return config
      }(),
      "INT4": {
        var config = Configuration()
        config.queryPrecision = .FP16
        config.keyPrecision = .INT4
        config.valuePrecision = .INT4
        return config
      }(),
    ]

    var results: [String: Double] = [:]

    for (name, config) in configs {
      let tensors = createQuantizedTensors(
        queryData: queryData, keyData: keyData, valueData: valueData,
        queryShape: shape, keyShape: shape, valueShape: shape,
        config: config
      )

      guard let outputBuffer = device.makeBuffer(length: totalElements * MemoryLayout<Float>.size)
      else {
        continue
      }

      var baseDescriptor = AttentionDescriptor()
      baseDescriptor.matrixDimensions = (
        row: UInt32(sequenceLength), column: UInt32(sequenceLength), head: UInt16(headDim)
      )
      baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

      let descriptor = QuantizedAttentionDescriptor(
        baseDescriptor: baseDescriptor,
        quantizationConfig: config
      )

      // Warmup - GPU kernels need extensive warmup to reach peak performance
      for _ in 0..<50 {
        if
          let commandBuffer = forward(
            query: tensors.query,
            key: tensors.key,
            value: tensors.value,
            output: outputBuffer,
            descriptor: descriptor
          )
        {
          commandBuffer.commit()
          commandBuffer.waitUntilCompleted()
        }
      }

      // Benchmark
      let startTime = CFAbsoluteTimeGetCurrent()
      for _ in 0..<iterations {
        if
          let commandBuffer = forward(
            query: tensors.query,
            key: tensors.key,
            value: tensors.value,
            output: outputBuffer,
            descriptor: descriptor
          )
        {
          commandBuffer.commit()
          commandBuffer.waitUntilCompleted()
        }
      }
      let endTime = CFAbsoluteTimeGetCurrent()

      let avgTime = (endTime - startTime) / Double(iterations)
      results[name + "_avg_ms"] = avgTime * 1000.0

      // Calculate GOPS
      let ops =
        2.0 * Double(batchSize) * Double(sequenceLength) * Double(sequenceLength) * Double(headDim)
      results[name + "_gops"] = ops / (avgTime * 1e9)
    }

    return results
  }
}

// MARK: - Quantized Backward Pass Implementation

extension QuantizedAttention {
  /// Perform quantized attention backward pass for query gradients.
  ///
  /// Dispatches the *same* core `AttentionKernel` (type `.backwardQuery`) that
  /// `SquareAttentionTest` validates against a CPU reference. INT8 operands are
  /// dequantized on load inside the kernel.
  ///
  /// - Parameters:
  ///   - query: Quantized query tensor
  ///   - key: Key operand (QuantizedTensor or raw MTLBuffer)
  ///   - value: Value operand (QuantizedTensor or raw MTLBuffer)
  ///   - output: Forward output O — required because the kernel computes D = dO·O
  ///   - gradOutput: Output gradients dO (FP32)
  ///   - logsumexp: Logsumexp L from the forward pass (FP32)
  ///   - gradQuery: Output buffer for dQ (FP32)
  ///   - dValues: Output buffer for the D intermediate (FP32)
  ///   - descriptor: Quantized attention configuration
  /// - Returns: Command buffer for execution
  public func backwardQuery(
    query: QuantizedTensor,
    key: Any,
    value: Any,
    output: MTLBuffer,
    gradOutput: MTLBuffer,
    logsumexp: MTLBuffer,
    gradQuery: MTLBuffer,
    dValues: MTLBuffer,
    descriptor: QuantizedAttentionDescriptor,
    bufferOffsets: (q: Int, k: Int, v: Int, o: Int, go: Int, lse: Int, gq: Int, dv: Int) = (0, 0, 0, 0, 0, 0, 0, 0)
  )
    -> MTLCommandBuffer?
  {
    guard
      !isDisposed, let queue = commandQueue,
      let commandBuffer = queue.makeCommandBuffer(),
      let keyBinding = makeBinding(for: key, label: "key"),
      let valueBinding = makeBinding(for: value, label: "value"),
      let dims = descriptor.baseDescriptor.matrixDimensions
    else {
      print("Error: Failed to set up backward query")
      return nil
    }

    // Detect blockwise quantization for function-constant selection.
    let bwQ = query.blockScales != nil && query.blockSizeK != nil
    let bwK = keyBinding.blockScales != nil && keyBinding.blockSize != nil
    let bwV = valueBinding.blockScales != nil && valueBinding.blockSize != nil
    let bsK = UInt32(
      query.blockSizeK ?? keyBinding.blockSize ?? valueBinding.blockSize ?? 1
    )

    guard
      let core = getOrCreateCorePipeline(
        type: .backwardQuery, descriptor: descriptor,
        hasBlockwiseQ: bwQ, hasBlockwiseK: bwK, hasBlockwiseV: bwV,
        blockSizeK: bsK
      ),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("Error: Failed to set up backward query pipeline")
      return nil
    }

    encoder.setComputePipelineState(core.pipeline)
    encoder.setThreadgroupMemoryLength(Int(core.kernel.threadgroupMemoryAllocation), index: 0)

    // Operands (AttentionKernel.createBufferBindings, single-head layout):
    //   Q@0 K@1 V@2 O@3 L@4 D@5 dO@6 ... dQ@9
    // Strides / multi-head / mask left unset → null → single-head mode.
    encoder.setBuffer(query.data, offset: bufferOffsets.q, index: 0)
    encoder.setBuffer(keyBinding.buffer, offset: bufferOffsets.k, index: 1)
    encoder.setBuffer(valueBinding.buffer, offset: bufferOffsets.v, index: 2)
    encoder.setBuffer(output, offset: bufferOffsets.o, index: 3)
    encoder.setBuffer(logsumexp, offset: bufferOffsets.lse, index: 4)
    encoder.setBuffer(dValues, offset: bufferOffsets.dv, index: 5)
    encoder.setBuffer(gradOutput, offset: bufferOffsets.go, index: 6)
    encoder.setBuffer(gradQuery, offset: bufferOffsets.gq, index: 9)

    // Per-tensor + blockwise quant params for quantized Q/K/V.
    bindQuantParams(
      encoder, query: query, key: keyBinding, value: valueBinding,
      config: descriptor.quantizationConfig, startingAt: 10
    )

    dispatchBackward(encoder, kernel: core.kernel, parallelizationDim: Int(dims.row))
    encoder.endEncoding()
    return commandBuffer
  }

  /// Perform quantized attention backward pass for key and value gradients.
  ///
  /// Dispatches the core `AttentionKernel` (type `.backwardKeyValue`).
  public func backwardKeyValue(
    query: QuantizedTensor,
    key: Any,
    value: Any,
    gradOutput: MTLBuffer,
    logsumexp: MTLBuffer,
    dValues: MTLBuffer,
    gradKey: MTLBuffer,
    gradValue: MTLBuffer,
    descriptor: QuantizedAttentionDescriptor,
    bufferOffsets: (q: Int, k: Int, v: Int, go: Int, lse: Int, dv: Int, gk: Int, gv: Int) = (0, 0, 0, 0, 0, 0, 0, 0)
  )
    -> MTLCommandBuffer?
  {
    guard
      !isDisposed, let queue = commandQueue,
      let commandBuffer = queue.makeCommandBuffer(),
      let keyBinding = makeBinding(for: key, label: "key"),
      let valueBinding = makeBinding(for: value, label: "value"),
      let dims = descriptor.baseDescriptor.matrixDimensions
    else {
      print("Error: Failed to set up backward key-value")
      return nil
    }

    // Detect blockwise quantization for function-constant selection.
    let bwQ = query.blockScales != nil && query.blockSizeK != nil
    let bwK = keyBinding.blockScales != nil && keyBinding.blockSize != nil
    let bwV = valueBinding.blockScales != nil && valueBinding.blockSize != nil
    let bsK = UInt32(
      query.blockSizeK ?? keyBinding.blockSize ?? valueBinding.blockSize ?? 1
    )

    guard
      let core = getOrCreateCorePipeline(
        type: .backwardKeyValue, descriptor: descriptor,
        hasBlockwiseQ: bwQ, hasBlockwiseK: bwK, hasBlockwiseV: bwV,
        blockSizeK: bsK
      ),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("Error: Failed to set up backward key-value pipeline")
      return nil
    }

    encoder.setComputePipelineState(core.pipeline)
    encoder.setThreadgroupMemoryLength(Int(core.kernel.threadgroupMemoryAllocation), index: 0)

    // Operands: Q@0 K@1 V@2 L@4 D@5 dO@6 dV@7 dK@8
    encoder.setBuffer(query.data, offset: bufferOffsets.q, index: 0)
    encoder.setBuffer(keyBinding.buffer, offset: bufferOffsets.k, index: 1)
    encoder.setBuffer(valueBinding.buffer, offset: bufferOffsets.v, index: 2)
    encoder.setBuffer(logsumexp, offset: bufferOffsets.lse, index: 4)
    encoder.setBuffer(dValues, offset: bufferOffsets.dv, index: 5)
    encoder.setBuffer(gradOutput, offset: bufferOffsets.go, index: 6)
    encoder.setBuffer(gradValue, offset: bufferOffsets.gv, index: 7)
    encoder.setBuffer(gradKey, offset: bufferOffsets.gk, index: 8)

    bindQuantParams(
      encoder, query: query, key: keyBinding, value: valueBinding,
      config: descriptor.quantizationConfig, startingAt: 9
    )

    dispatchBackward(encoder, kernel: core.kernel, parallelizationDim: Int(dims.row))
    encoder.endEncoding()
    return commandBuffer
  }

  // MARK: - Core backward pipeline helpers

  /// Build the proven core `AttentionKernel` pipeline for the given type.
  /// Supports both per-tensor and blockwise quantization: `HAS_BLOCKWISE_*`
  /// function constants are set from the actual operand properties.
  private func getOrCreateCorePipeline(
    type: AttentionKernelType,
    descriptor: QuantizedAttentionDescriptor,
    hasBlockwiseQ: Bool = false,
    hasBlockwiseK: Bool = false,
    hasBlockwiseV: Bool = false,
    blockSizeK: UInt32 = 1
  )
    -> (pipeline: MTLComputePipelineState, kernel: AttentionKernel)?
  {
    let kernel = AttentionKernel(descriptor: descriptor.kernelDescriptor(type: type))
    let source = kernel.createSource()
    let cacheKey =
      "core_\(type)_\(source.hashValue)_\(hasBlockwiseQ ? 1 : 0)\(hasBlockwiseK ? 1 : 0)\(hasBlockwiseV ? 1 : 0)_\(blockSizeK)"

    if let cached = pipelineCache[cacheKey] {
      return (cached, kernel)
    }

    do {
      let library = try device.makeLibrary(source: source, options: nil)
      let functionConstants = MTLFunctionConstantValues()
      descriptor.baseDescriptor.setFunctionConstants(functionConstants)
      var bq = hasBlockwiseQ
      var bk = hasBlockwiseK
      var bv = hasBlockwiseV
      var blockSize = blockSizeK
      functionConstants.setConstantValue(&bq, type: .bool, index: 5)
      functionConstants.setConstantValue(&bk, type: .bool, index: 6)
      functionConstants.setConstantValue(&bv, type: .bool, index: 7)
      functionConstants.setConstantValue(&blockSize, type: .uint, index: 8)

      let function = try library.makeFunction(name: "attention", constantValues: functionConstants)
      let pipelineState = try device.makeComputePipelineState(function: function)
      pipelineCache[cacheKey] = pipelineState
      return (pipelineState, kernel)
    } catch {
      print("Pipeline creation error (\(type)): \(error)")
      return nil
    }
  }

  /// Bind per-tensor (scale, zero_point) for the quantized operands among
  /// Q/K/V, in bufferBinding order — mirroring `AttentionKernel.createBufferBindings`
  /// pass 2. Then bind blockwise buffers (pass 3): block_scales and
  /// block_zero_points per quantized operand.
  private func bindQuantParams(
    _ encoder: MTLComputeCommandEncoder,
    query: QuantizedTensor,
    key: OperandBinding,
    value: OperandBinding,
    config: QuantizedAttention.Configuration,
    startingAt start: Int
  ) {
    var index = start

    // Pass 2: per-tensor scale and zero_point.
    func emitScaleZeroPoint(_ scale: Float, _ zeroPoint: Int32) {
      var scale = scale
      encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: index)
      index += 1
      var zeroPoint = zeroPoint
      encoder.setBytes(&zeroPoint, length: MemoryLayout<Int32>.size, index: index)
      index += 1
    }
    if config.queryPrecision.requiresQuantizationParameters {
      emitScaleZeroPoint(query.parameters.scale, Int32(query.parameters.zeroPoint))
    }
    if config.keyPrecision.requiresQuantizationParameters {
      emitScaleZeroPoint(key.scale, key.zeroPoint)
    }
    if config.valuePrecision.requiresQuantizationParameters {
      emitScaleZeroPoint(value.scale, value.zeroPoint)
    }

    // Pass 3: blockwise scales and zero_points (may be nil → per-tensor fallback).
    func emitBlockBuffers(_ scales: MTLBuffer?, _ zeroPoints: MTLBuffer?) {
      encoder.setBuffer(scales, offset: 0, index: index)
      index += 1
      encoder.setBuffer(zeroPoints, offset: 0, index: index)
      index += 1
    }
    if config.queryPrecision.requiresQuantizationParameters {
      emitBlockBuffers(query.blockScales, query.blockZeroPoints)
    }
    if config.keyPrecision.requiresQuantizationParameters {
      emitBlockBuffers(key.blockScales, key.blockZeroPoints)
    }
    if config.valuePrecision.requiresQuantizationParameters {
      emitBlockBuffers(value.blockScales, value.blockZeroPoints)
    }
  }

  private func dispatchBackward(
    _ encoder: MTLComputeCommandEncoder,
    kernel: AttentionKernel,
    parallelizationDim: Int
  ) {
    let blockParallelization = Int(kernel.blockDimensions.parallelization)
    let blockCount = (parallelizationDim + blockParallelization - 1) / blockParallelization
    let threadgroupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)
    let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
  }

  // MARK: - Private Helper Methods

  private struct PrecisionConfig {
    let keyQuantized: Bool
    let valueQuantized: Bool
  }

  private struct OperandBinding {
    let buffer: MTLBuffer
    let scale: Float
    let zeroPoint: Int32
    let precision: GEMMOperandPrecision
    let shape: [Int]?
    let blockScales: MTLBuffer?
    let blockZeroPoints: MTLBuffer?
    let precomputedSums: MTLBuffer?
    let blockSize: Int?
  }

  private func detectPrecisionConfiguration(
    keyBinding: OperandBinding,
    valueBinding: OperandBinding
  )
    -> PrecisionConfig
  {
    PrecisionConfig(
      keyQuantized: keyBinding.precision.requiresQuantizationParameters,
      valueQuantized: valueBinding.precision.requiresQuantizationParameters
    )
  }

  private func makeBinding(for operand: Any, label: String) -> OperandBinding? {
    if let tensor = operand as? QuantizedTensor {
      let zeroPoint = Int32(tensor.parameters.zeroPoint)
      return OperandBinding(
        buffer: tensor.data,
        scale: tensor.parameters.scale,
        zeroPoint: zeroPoint,
        precision: tensor.parameters.precision,
        shape: tensor.originalShape,
        blockScales: tensor.blockScales,
        blockZeroPoints: tensor.blockZeroPoints,
        precomputedSums: tensor.precomputedSums,
        blockSize: tensor.blockSizeK
      )
    }

    if let buffer = operand as? MTLBuffer {
      return OperandBinding(
        buffer: buffer,
        scale: 1.0,
        zeroPoint: 0,
        precision: .FP16,
        shape: nil,
        blockScales: nil,
        blockZeroPoints: nil,
        precomputedSums: nil,
        blockSize: nil
      )
    }

    print("Error: Unsupported \(label) operand type: \(String(describing: type(of: operand)))")
    return nil
  }

  private func setBufferIfValid(
    _ encoder: MTLComputeCommandEncoder,
    buffer: MTLBuffer?,
    index: Int
  ) {
    guard index >= 0 else { return }
    encoder.setBuffer(buffer, offset: 0, index: index)
  }

  private func setBytesIfValid(
    _ encoder: MTLComputeCommandEncoder,
    value: inout some Any,
    index: Int
  ) {
    guard index >= 0 else { return }
    withUnsafeBytes(of: &value) { rawBuffer in
      encoder.setBytes(rawBuffer.baseAddress!, length: rawBuffer.count, index: index)
    }
  }

  private func setArrayIfValid(
    _ encoder: MTLComputeCommandEncoder,
    values: inout [Int64],
    index: Int
  ) {
    guard index >= 0, !values.isEmpty else { return }
    values.withUnsafeBytes { rawBuffer in
      encoder.setBytes(rawBuffer.baseAddress!, length: rawBuffer.count, index: index)
    }
  }

  private func normalizedShape(_ shape: [Int]?, rank: Int = 4) -> [Int] {
    guard let shape else {
      return Array(repeating: 1, count: rank)
    }
    if shape.count >= rank {
      return Array(shape.suffix(rank))
    }
    let padding = Array(repeating: 1, count: rank - shape.count)
    return padding + shape
  }

  private struct TensorShapeInfo {
    let batch: Int
    let heads: Int
    let sequence: Int
    let headDimension: Int
  }

  private struct MultiHeadMetadata {
    let qShape: TensorShapeInfo
    let kShape: TensorShapeInfo
    let vShape: TensorShapeInfo
    let numHeads: UInt32
    let numKVHeads: UInt32
    let qStrides: [Int64]
    let kStrides: [Int64]
    let vStrides: [Int64]
    let oStrides: [Int64]
    let sequenceLengthQ: UInt32
  }

  private func decodeTensorShape(
    _ shape: [Int]?,
    fallbackBatch: Int,
    fallbackHeads: Int,
    fallbackSequence: Int,
    fallbackHeadDimension: Int
  )
    -> TensorShapeInfo
  {
    if let shape {
      if shape.count >= 4 {
        let base = Array(shape.suffix(4))
        return TensorShapeInfo(
          batch: max(base[0], 1),
          heads: max(base[1], 1),
          sequence: max(base[2], fallbackSequence),
          headDimension: max(base[3], fallbackHeadDimension)
        )
      } else if shape.count == 3 {
        return TensorShapeInfo(
          batch: max(shape[0], 1),
          heads: max(fallbackHeads, 1),
          sequence: max(shape[1], fallbackSequence),
          headDimension: max(shape[2], fallbackHeadDimension)
        )
      } else if shape.count == 2 {
        return TensorShapeInfo(
          batch: max(fallbackBatch, 1),
          heads: max(fallbackHeads, 1),
          sequence: max(shape[0], fallbackSequence),
          headDimension: max(shape[1], fallbackHeadDimension)
        )
      }
    }

    return TensorShapeInfo(
      batch: max(fallbackBatch, 1),
      heads: max(fallbackHeads, 1),
      sequence: max(fallbackSequence, 1),
      headDimension: max(fallbackHeadDimension, 1)
    )
  }

  private func prepareMultiHeadMetadata(
    query: QuantizedTensor,
    keyBinding: OperandBinding,
    valueBinding: OperandBinding,
    descriptor _: QuantizedAttentionDescriptor,
    dims: (row: UInt32, column: UInt32, head: UInt16)
  )
    -> MultiHeadMetadata
  {
    let fallbackHeadDim = max(Int(dims.head), 1)
    let fallbackSequenceQ = max(Int(dims.row), 1)
    let fallbackSequenceKV = max(Int(dims.column), 1)

    let denominator = max(fallbackSequenceQ * fallbackHeadDim, 1)
    let fallbackBatch = max(query.elementCount / denominator, 1)

    let qShape = decodeTensorShape(
      query.originalShape,
      fallbackBatch: fallbackBatch,
      fallbackHeads: 1,
      fallbackSequence: fallbackSequenceQ,
      fallbackHeadDimension: fallbackHeadDim
    )

    let kShape = decodeTensorShape(
      keyBinding.shape,
      fallbackBatch: qShape.batch,
      fallbackHeads: max(qShape.heads, 1),
      fallbackSequence: fallbackSequenceKV,
      fallbackHeadDimension: fallbackHeadDim
    )

    let vShape = decodeTensorShape(
      valueBinding.shape,
      fallbackBatch: qShape.batch,
      fallbackHeads: max(kShape.heads, 1),
      fallbackSequence: fallbackSequenceKV,
      fallbackHeadDimension: fallbackHeadDim
    )

    let qShapeArray = [qShape.batch, qShape.heads, qShape.sequence, qShape.headDimension]
    let kShapeArray = [kShape.batch, kShape.heads, kShape.sequence, kShape.headDimension]
    let vShapeArray = [vShape.batch, vShape.heads, vShape.sequence, vShape.headDimension]

    let qStrides = contiguousStrides(
      shape: qShapeArray,
      elementSize: query.parameters.precision.size
    )
    let kStrides = contiguousStrides(shape: kShapeArray, elementSize: keyBinding.precision.size)
    let vStrides = contiguousStrides(shape: vShapeArray, elementSize: valueBinding.precision.size)
    let oStrides = contiguousStrides(shape: qShapeArray, elementSize: MemoryLayout<Float>.size)

    let numHeads = UInt32(max(qShape.heads, 1))
    let numKVHeads = UInt32(max(kShape.heads, 1))

    return MultiHeadMetadata(
      qShape: qShape,
      kShape: kShape,
      vShape: vShape,
      numHeads: numHeads,
      numKVHeads: numKVHeads,
      qStrides: qStrides,
      kStrides: kStrides,
      vStrides: vStrides,
      oStrides: oStrides,
      sequenceLengthQ: UInt32(max(qShape.sequence, 1))
    )
  }

  private func contiguousStrides(shape: [Int], elementSize: Int) -> [Int64] {
    var strides = Array(repeating: Int64(0), count: shape.count)
    var running = Int64(elementSize)
    for index in stride(from: shape.count - 1, through: 0, by: -1) {
      strides[index] = running
      running *= Int64(shape[index])
    }
    return strides
  }

  // STE (straight-through estimator) backward generators were removed.
  //
  // The previous implementation (generateQuantizedBackwardQueryKernel /
  // generateQuantizedBackwardKeyValueKernel) used a naive non-flash backward
  // with hand-rolled attention weights, soft clipping, and damping factors.
  // They were never called — the actual backward path dispatches through the
  // core flash-attention kernel via getOrCreateCorePipeline(.backwardQuery /
  // .backwardKeyValue), which is exact and leverages the proven flash
  // algorithm with dequantize-on-load.
  //
  // STE for quantization rounding is handled at the C++ autograd level
  // (MetalFlashAttentionFn in metal_sdpa_backend.cpp): the forward pass
  // quantizes Q/K/V (fake-quant), and the backward pass passes gradients
  // straight through the rounding step (STE). This is cleaner and keeps the
  // Metal kernel focused on the flash-attention math.
}

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

    /// Whether to use mixed precision intermediate computations
    public var mixedPrecisionIntermediates: Bool = true

    /// Quantization parameters for each tensor
    public var quantizationParameters: [String: QuantizationParameters] = [:]

    public init() {}
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
  private let commandQueue: MTLCommandQueue
  private var pipelineCache: [String: MTLComputePipelineState] = [:]

  public init(device: MTLDevice) {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      fatalError("Could not create Metal command queue")
    }
    commandQueue = queue
  }

  /// Perform quantized attention forward pass
  /// - Parameters:
  ///   - query: Query tensor (can be FP32, FP16, or quantized)
  ///   - key: Key tensor (can be FP32, FP16, or quantized)
  ///   - value: Value tensor (can be FP32, FP16, or quantized)
  ///   - output: Output tensor buffer
  ///   - descriptor: Quantized attention configuration
  /// - Returns: Command buffer for execution
  public func forward(
    query: QuantizedTensor,
    key: QuantizedTensor,
    value: QuantizedTensor,
    output: MTLBuffer,
    descriptor: QuantizedAttentionDescriptor
  )
    -> MTLCommandBuffer?
  {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      print("Error: Failed to create command buffer")
      return nil
    }

    let kernelDescriptor = descriptor.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDescriptor)

    // Create pipeline state for quantized attention
    guard let pipelineState = getOrCreatePipelineState(for: kernel, descriptor: descriptor) else {
      print("Error: Failed to create pipeline state")
      return nil
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    encoder.setComputePipelineState(pipelineState)

    // Set tensor buffers
    encoder.setBuffer(query.data, offset: 0, index: 0)
    encoder.setBuffer(key.data, offset: 0, index: 1)
    encoder.setBuffer(value.data, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)

    // Set quantization parameters
    var bufferIndex = 4

    if query.parameters.precision.requiresQuantizationParameters {
      var qScale = query.parameters.scale
      var qZeroPoint = query.parameters.zeroPoint
      encoder.setBytes(&qScale, length: MemoryLayout<Float>.size, index: bufferIndex)
      encoder.setBytes(&qZeroPoint, length: MemoryLayout<Int32>.size, index: bufferIndex + 1)
      bufferIndex += 2
    }

    if key.parameters.precision.requiresQuantizationParameters {
      var kScale = key.parameters.scale
      var kZeroPoint = key.parameters.zeroPoint
      encoder.setBytes(&kScale, length: MemoryLayout<Float>.size, index: bufferIndex)
      encoder.setBytes(&kZeroPoint, length: MemoryLayout<Int32>.size, index: bufferIndex + 1)
      bufferIndex += 2
    }

    if value.parameters.precision.requiresQuantizationParameters {
      var vScale = value.parameters.scale
      var vZeroPoint = value.parameters.zeroPoint
      encoder.setBytes(&vScale, length: MemoryLayout<Float>.size, index: bufferIndex)
      encoder.setBytes(&vZeroPoint, length: MemoryLayout<Int32>.size, index: bufferIndex + 1)
      bufferIndex += 2
    }

    // Set matrix dimensions
    let dims = descriptor.baseDescriptor.matrixDimensions!
    var M = UInt32(dims.row)
    var N = UInt32(dims.column)
    var K = UInt32(dims.head)

    encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: bufferIndex)
    encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: bufferIndex + 1)
    encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: bufferIndex + 2)

    // Calculate optimal thread group size for GPU matrix operations
    let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1) // GPU-friendly tile size
    let gridSize = MTLSize(
      width: (Int(N) + threadgroupSize.width - 1) / threadgroupSize.width,
      height: (Int(M) + threadgroupSize.height - 1) / threadgroupSize.height,
      depth: 1
    )

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    return commandBuffer
  }

  private func getOrCreatePipelineState(
    for kernel: AttentionKernel, descriptor: QuantizedAttentionDescriptor
  )
    -> MTLComputePipelineState?
  {
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
}

// MARK: - Convenience extensions

public extension QuantizedAttention {
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
      precision: config.queryPrecision
    )

    let key = QuantizedTensor.from(
      device: device,
      floatData: keyData,
      shape: keyShape,
      precision: config.keyPrecision
    )

    let value = QuantizedTensor.from(
      device: device,
      floatData: valueData,
      shape: valueShape,
      precision: config.valuePrecision
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
  /// Perform quantized attention backward pass for query gradients
  /// - Parameters:
  ///   - query: Quantized query tensor (INT8)
  ///   - key: Key tensor (can be FP16 or quantized)
  ///   - value: Value tensor (can be FP16 or quantized)
  ///   - gradOutput: Output gradients (FP32)
  ///   - logsumexp: Logsumexp values from forward pass (FP32)
  ///   - gradQuery: Output buffer for query gradients (FP32)
  ///   - dValues: Output buffer for D intermediate values (FP32)
  ///   - descriptor: Quantized attention configuration
  /// - Returns: Command buffer for execution
  public func backwardQuery(
    query: QuantizedTensor,
    key: QuantizedTensor,
    value: QuantizedTensor,
    gradOutput: MTLBuffer,
    logsumexp: MTLBuffer,
    gradQuery: MTLBuffer,
    dValues: MTLBuffer,
    descriptor: QuantizedAttentionDescriptor
  )
    -> MTLCommandBuffer?
  {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      print("Error: Failed to create command buffer for backward query")
      return nil
    }

    // Generate quantized backward query kernel
    let source = generateQuantizedBackwardQueryKernel(descriptor: descriptor)

    guard
      let pipelineState = createQuantizedBackwardPipeline(
        source: source, functionName: "quantized_backward_query"
      )
    else {
      print("Error: Failed to create pipeline state for backward query")
      return nil
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    encoder.setComputePipelineState(pipelineState)

    // Set buffers
    encoder.setBuffer(query.data, offset: 0, index: 0) // Q_quantized
    encoder.setBuffer(key.data, offset: 0, index: 1) // K
    encoder.setBuffer(value.data, offset: 0, index: 2) // V
    encoder.setBuffer(gradOutput, offset: 0, index: 3) // dO
    encoder.setBuffer(logsumexp, offset: 0, index: 4) // L
    encoder.setBuffer(gradQuery, offset: 0, index: 5) // dQ
    encoder.setBuffer(dValues, offset: 0, index: 6) // D

    // Set quantization parameters
    var qScale = query.parameters.scale
    var qZeroPoint = query.parameters.zeroPoint
    encoder.setBytes(&qScale, length: MemoryLayout<Float>.size, index: 7)
    encoder.setBytes(&qZeroPoint, length: MemoryLayout<Int32>.size, index: 8)

    // Set matrix dimensions
    let dims = descriptor.baseDescriptor.matrixDimensions!
    var dimensions = (UInt32(dims.row), UInt32(dims.column), UInt32(dims.head))
    encoder.setBytes(&dimensions, length: MemoryLayout.size(ofValue: dimensions), index: 9)

    // Set STE clip range
    var steClipRange: Float = 6.0
    encoder.setBytes(&steClipRange, length: MemoryLayout<Float>.size, index: 10)

    // Calculate thread groups
    let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
    let gridSize = MTLSize(
      width: (Int(dims.head) + threadgroupSize.width - 1) / threadgroupSize.width,
      height: (Int(dims.row) + threadgroupSize.height - 1) / threadgroupSize.height,
      depth: 1
    )

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    return commandBuffer
  }

  /// Perform quantized attention backward pass for key and value gradients
  /// - Parameters:
  ///   - query: Quantized query tensor (INT8)
  ///   - key: Quantized key tensor (INT8)
  ///   - value: Quantized value tensor (INT8)
  ///   - gradOutput: Output gradients (FP32)
  ///   - logsumexp: Logsumexp values from forward pass (FP32)
  ///   - dValues: D intermediate values from backward query (FP32)
  ///   - gradKey: Output buffer for key gradients (FP32)
  ///   - gradValue: Output buffer for value gradients (FP32)
  ///   - descriptor: Quantized attention configuration
  /// - Returns: Command buffer for execution
  public func backwardKeyValue(
    query: QuantizedTensor,
    key: QuantizedTensor,
    value: QuantizedTensor,
    gradOutput: MTLBuffer,
    logsumexp: MTLBuffer,
    dValues: MTLBuffer,
    gradKey: MTLBuffer,
    gradValue: MTLBuffer,
    descriptor: QuantizedAttentionDescriptor
  )
    -> MTLCommandBuffer?
  {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      print("Error: Failed to create command buffer for backward key-value")
      return nil
    }

    // Generate quantized backward key-value kernel
    let source = generateQuantizedBackwardKeyValueKernel(descriptor: descriptor)

    guard
      let pipelineState = createQuantizedBackwardPipeline(
        source: source, functionName: "quantized_backward_key_value"
      )
    else {
      print("Error: Failed to create pipeline state for backward key-value")
      return nil
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return nil
    }

    encoder.setComputePipelineState(pipelineState)

    // Set buffers
    encoder.setBuffer(query.data, offset: 0, index: 0) // Q_quantized
    encoder.setBuffer(key.data, offset: 0, index: 1) // K_quantized
    encoder.setBuffer(value.data, offset: 0, index: 2) // V_quantized
    encoder.setBuffer(gradOutput, offset: 0, index: 3) // dO
    encoder.setBuffer(logsumexp, offset: 0, index: 4) // L
    encoder.setBuffer(dValues, offset: 0, index: 5) // D
    encoder.setBuffer(gradKey, offset: 0, index: 6) // dK
    encoder.setBuffer(gradValue, offset: 0, index: 7) // dV

    // Set quantization parameters for Q, K, V
    var qScale = query.parameters.scale
    var qZeroPoint = query.parameters.zeroPoint
    var kScale = key.parameters.scale
    var kZeroPoint = key.parameters.zeroPoint
    var vScale = value.parameters.scale
    var vZeroPoint = value.parameters.zeroPoint

    encoder.setBytes(&qScale, length: MemoryLayout<Float>.size, index: 8)
    encoder.setBytes(&qZeroPoint, length: MemoryLayout<Int32>.size, index: 9)
    encoder.setBytes(&kScale, length: MemoryLayout<Float>.size, index: 10)
    encoder.setBytes(&kZeroPoint, length: MemoryLayout<Int32>.size, index: 11)
    encoder.setBytes(&vScale, length: MemoryLayout<Float>.size, index: 12)
    encoder.setBytes(&vZeroPoint, length: MemoryLayout<Int32>.size, index: 13)

    // Set matrix dimensions
    let dims = descriptor.baseDescriptor.matrixDimensions!
    var dimensions = (UInt32(dims.row), UInt32(dims.column), UInt32(dims.head))
    encoder.setBytes(&dimensions, length: MemoryLayout.size(ofValue: dimensions), index: 14)

    // Set STE clip range
    var steClipRange: Float = 6.0
    encoder.setBytes(&steClipRange, length: MemoryLayout<Float>.size, index: 15)

    // Calculate thread groups
    let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
    let gridSize = MTLSize(
      width: (Int(dims.head) + threadgroupSize.width - 1) / threadgroupSize.width,
      height: (Int(dims.column) + threadgroupSize.height - 1) / threadgroupSize.height,
      depth: 1
    )

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()

    return commandBuffer
  }

  // MARK: - Private Helper Methods

  private func createQuantizedBackwardPipeline(source: String, functionName: String)
    -> MTLComputePipelineState?
  {
    let cacheKey = "\(functionName)_\(source.hashValue)"

    if let cached = pipelineCache[cacheKey] {
      return cached
    }

    do {
      let library = try device.makeLibrary(source: source, options: nil)
      guard let function = library.makeFunction(name: functionName) else {
        print("Error: Function '\(functionName)' not found in library")
        return nil
      }
      let pipelineState = try device.makeComputePipelineState(function: function)

      pipelineCache[cacheKey] = pipelineState
      return pipelineState
    } catch {
      print("Pipeline creation error for \(functionName): \(error)")
      return nil
    }
  }

  private func generateQuantizedBackwardQueryKernel(descriptor _: QuantizedAttentionDescriptor)
    -> String
  {
    """
    #include <metal_stdlib>
    using namespace metal;

    // Vectorized dequantization helper
    METAL_FUNC float4 dequantize_char4(char4 quantized, float scale, int32_t zero_point) {
        int4 int_vals = int4(quantized);
        return (float4(int_vals) - float(zero_point)) * scale;
    }

    // Quantized backward pass: compute dQ
    kernel void quantized_backward_query(
        device const char *Q_quantized [[buffer(0)]],      // INT8 quantized query
        device const half *K [[buffer(1)]],                // FP16 key
        device const half *V [[buffer(2)]],                // FP16 value
        device const float *dO [[buffer(3)]],              // FP32 output gradients
        device const float *L [[buffer(4)]],               // FP32 logsumexp from forward
        device float *dQ [[buffer(5)]],                    // FP32 query gradients
        device float *D [[buffer(6)]],                     // FP32 intermediate D values
        constant float &q_scale [[buffer(7)]],
        constant int32_t &q_zero_point [[buffer(8)]],
        constant uint3 &dims [[buffer(9)]],                // {M, N, K}
        constant float &ste_clip_range [[buffer(10)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint M = dims.x, N = dims.y, K = dims.z;
        uint row = gid.y, col = gid.x;

        if (row >= M || col >= K) return;

        // Dequantize current query element
        uint q_idx = row * K + col;
        char q_quantized = Q_quantized[q_idx];
        float q_dequantized = (float(q_quantized) - float(q_zero_point)) * q_scale;

        // Clipped straight-through estimator
        float ste_gradient = 1.0f;
        if (abs(q_dequantized) > ste_clip_range) {
            ste_gradient = 0.0f;  // Zero gradient outside quantization range
        }

        // Compute D value (rowsum of dO * O) - simplified for prototype
        if (col == 0) {
            float d_val = 0.0f;
            for (uint k = 0; k < K; k++) {
                d_val += dO[row * K + k];  // Simplified: assume O ≈ 1 for prototype
            }
            D[row] = d_val;
        }

        // Attention backward computation: dQ = (P - diag(D)) * K
        float dq_accumulator = 0.0f;

        for (uint n = 0; n < N; n++) {
            // Compute QK^T dot product
            float qk_dot = 0.0f;
            for (uint k = 0; k < K; k++) {
                float q_val = (float(Q_quantized[row * K + k]) - float(q_zero_point)) * q_scale;
                qk_dot += q_val * float(K[n * K + k]);
            }

            // Compute attention weight P[row, n]
            float p_val = exp(qk_dot - L[row]);

            // Apply diagonal correction for gradient computation
            if (row == n) {
                p_val -= D[row];
            }

            // Accumulate gradient: dQ += (P - diag(D)) * K
            dq_accumulator += p_val * float(K[n * K + col]);
        }

        // Apply straight-through estimator and store
        dQ[row * K + col] = dq_accumulator * ste_gradient;
    }
    """
  }

  private func generateQuantizedBackwardKeyValueKernel(descriptor _: QuantizedAttentionDescriptor)
    -> String
  {
    """
    #include <metal_stdlib>
    using namespace metal;

    // Vectorized dequantization helper
    METAL_FUNC float4 dequantize_char4(char4 quantized, float scale, int32_t zero_point) {
        int4 int_vals = int4(quantized);
        return (float4(int_vals) - float(zero_point)) * scale;
    }

    // Quantized backward pass: compute dK and dV
    kernel void quantized_backward_key_value(
        device const char *Q_quantized [[buffer(0)]],      // INT8 quantized query
        device const char *K_quantized [[buffer(1)]],      // INT8 quantized key
        device const char *V_quantized [[buffer(2)]],      // INT8 quantized value
        device const float *dO [[buffer(3)]],              // FP32 output gradients
        device const float *L [[buffer(4)]],               // FP32 logsumexp from forward
        device const float *D [[buffer(5)]],               // FP32 D values from backward_query
        device float *dK [[buffer(6)]],                    // FP32 key gradients
        device float *dV [[buffer(7)]],                    // FP32 value gradients
        constant float &q_scale [[buffer(8)]],
        constant int32_t &q_zero_point [[buffer(9)]],
        constant float &k_scale [[buffer(10)]],
        constant int32_t &k_zero_point [[buffer(11)]],
        constant float &v_scale [[buffer(12)]],
        constant int32_t &v_zero_point [[buffer(13)]],
        constant uint3 &dims [[buffer(14)]],               // {M, N, K}
        constant float &ste_clip_range [[buffer(15)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint M = dims.x, N = dims.y, K = dims.z;
        uint row = gid.y, col = gid.x;

        if (row >= N || col >= K) return;

        // Dequantize current K and V elements
        char k_quantized = K_quantized[row * K + col];
        char v_quantized = V_quantized[row * K + col];

        float k_dequantized = (float(k_quantized) - float(k_zero_point)) * k_scale;
        float v_dequantized = (float(v_quantized) - float(v_zero_point)) * v_scale;

        // Clipped straight-through estimators
        float k_ste = (abs(k_dequantized) <= ste_clip_range) ? 1.0f : 0.0f;
        float v_ste = (abs(v_dequantized) <= ste_clip_range) ? 1.0f : 0.0f;

        // Compute dK and dV
        float dk_accumulator = 0.0f;
        float dv_accumulator = 0.0f;

        for (uint m = 0; m < M; m++) {
            // Compute QK^T dot product for attention weight
            float qk_dot = 0.0f;
            for (uint k = 0; k < K; k++) {
                float q_k = (float(Q_quantized[m * K + k]) - float(q_zero_point)) * q_scale;
                float k_k = (float(K_quantized[row * K + k]) - float(k_zero_point)) * k_scale;
                qk_dot += q_k * k_k;
            }

            // Compute attention weight P[m, row]
            float p_val = exp(qk_dot - L[m]);

            // dK computation: Q^T * (P * dO - diag(D) * dO)
            float q_val = (float(Q_quantized[m * K + col]) - float(q_zero_point)) * q_scale;
            float p_do_correction = p_val * dO[m * K + col];
            if (m == row) { // Diagonal term
                p_do_correction -= D[m] * dO[m * K + col];
            }
            dk_accumulator += q_val * p_do_correction;

            // dV computation: P^T * dO
            dv_accumulator += p_val * dO[m * K + col];
        }

        // Apply straight-through estimators and store
        dK[row * K + col] = dk_accumulator * k_ste;
        dV[row * K + col] = dv_accumulator * v_ste;
    }
    """
  }
}

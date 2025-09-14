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
        self.commandQueue = queue
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
    ) -> MTLCommandBuffer? {

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
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)  // GPU-friendly tile size
        let gridSize = MTLSize(
            width: (Int(N) + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (Int(M) + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        return commandBuffer
    }

    private func getOrCreatePipelineState(for kernel: AttentionKernel, descriptor: QuantizedAttentionDescriptor) -> MTLComputePipelineState? {
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
    ) -> (query: QuantizedTensor, key: QuantizedTensor, value: QuantizedTensor) {

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
    ) -> [String: Double] {

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
            }()
        ]

        var results: [String: Double] = [:]

        for (name, config) in configs {
            let tensors = createQuantizedTensors(
                queryData: queryData, keyData: keyData, valueData: valueData,
                queryShape: shape, keyShape: shape, valueShape: shape,
                config: config
            )

            guard let outputBuffer = device.makeBuffer(length: totalElements * MemoryLayout<Float>.size) else {
                continue
            }

            var baseDescriptor = AttentionDescriptor()
            baseDescriptor.matrixDimensions = (row: UInt32(sequenceLength), column: UInt32(sequenceLength), head: UInt16(headDim))
            baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

            let descriptor = QuantizedAttentionDescriptor(
                baseDescriptor: baseDescriptor,
                quantizationConfig: config
            )

            // Warmup
            for _ in 0..<10 {
                if let commandBuffer = forward(
                    query: tensors.query,
                    key: tensors.key,
                    value: tensors.value,
                    output: outputBuffer,
                    descriptor: descriptor
                ) {
                    commandBuffer.commit()
                    commandBuffer.waitUntilCompleted()
                }
            }

            // Benchmark
            let startTime = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                if let commandBuffer = forward(
                    query: tensors.query,
                    key: tensors.key,
                    value: tensors.value,
                    output: outputBuffer,
                    descriptor: descriptor
                ) {
                    commandBuffer.commit()
                    commandBuffer.waitUntilCompleted()
                }
            }
            let endTime = CFAbsoluteTimeGetCurrent()

            let avgTime = (endTime - startTime) / Double(iterations)
            results[name + "_avg_ms"] = avgTime * 1000.0

            // Calculate GOPS
            let ops = 2.0 * Double(batchSize) * Double(sequenceLength) * Double(sequenceLength) * Double(headDim)
            results[name + "_gops"] = ops / (avgTime * 1e9)
        }

        return results
    }
}
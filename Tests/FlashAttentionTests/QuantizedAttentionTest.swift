//
//  QuantizedAttentionTest.swift
//  FlashAttentionTests
//
//

import XCTest
import Metal
@testable import FlashAttention

final class QuantizedAttentionTest: XCTestCase {

    var device: MTLDevice!
    var quantizedAttention: QuantizedAttention!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal is not supported on this device")
        quantizedAttention = QuantizedAttention(device: device)
    }

    override func tearDown() {
        quantizedAttention = nil
        device = nil
        super.tearDown()
    }

    func testQuantizationParameters() {
        // Test INT8 quantization parameter calculation
        let testData: [Float] = [-10.0, -5.0, 0.0, 5.0, 10.0]
        testData.withUnsafeBufferPointer { buffer in
            let params = GEMMOperandPrecision.INT8.calculateQuantizationParameters(
                data: buffer.baseAddress!,
                count: buffer.count
            )

            XCTAssertEqual(params.precision, .INT8)
            XCTAssertEqual(params.zeroPoint, 0)  // Symmetric quantization
            XCTAssertEqual(params.scale, 10.0 / 127.0, accuracy: 1e-6)
        }

        // Test INT4 quantization parameter calculation
        testData.withUnsafeBufferPointer { buffer in
            let params = GEMMOperandPrecision.INT4.calculateQuantizationParameters(
                data: buffer.baseAddress!,
                count: buffer.count
            )

            XCTAssertEqual(params.precision, .INT4)
            XCTAssertEqual(params.zeroPoint, 0)  // Symmetric quantization
            XCTAssertEqual(params.scale, 10.0 / 7.0, accuracy: 1e-6)
        }
    }

    func testQuantizeAndDequantize() {
        let originalData: [Float] = Array(stride(from: -10.0, through: 10.0, by: 0.5))
        let count = originalData.count

        // Test INT8 round-trip
        do {
            let params = GEMMOperandPrecision.INT8.calculateQuantizationParameters(
                data: originalData,
                count: count
            )

            var quantizedData = [Int8](repeating: 0, count: count)
            var dequantizedData = [Float](repeating: 0, count: count)

            originalData.withUnsafeBufferPointer { inputPtr in
                quantizedData.withUnsafeMutableBufferPointer { quantizedPtr in
                    GEMMOperandPrecision.INT8.quantize(
                        input: inputPtr.baseAddress!,
                        output: UnsafeMutableRawPointer(quantizedPtr.baseAddress!),
                        count: count,
                        parameters: params
                    )
                }
            }

            quantizedData.withUnsafeBufferPointer { quantizedPtr in
                dequantizedData.withUnsafeMutableBufferPointer { dequantizedPtr in
                    GEMMOperandPrecision.INT8.dequantize(
                        input: UnsafeRawPointer(quantizedPtr.baseAddress!),
                        output: dequantizedPtr.baseAddress!,
                        count: count,
                        parameters: params
                    )
                }
            }

            // Check that dequantized values are close to original
            for i in 0..<count {
                let error = abs(dequantizedData[i] - originalData[i])
                let tolerance = params.scale * 2  // Allow for quantization error
                XCTAssertLessThan(error, tolerance,
                    "INT8 quantization error too large at index \(i): \(error) > \(tolerance)")
            }
        }

        // Test INT4 round-trip
        do {
            let params = GEMMOperandPrecision.INT4.calculateQuantizationParameters(
                data: originalData,
                count: count
            )

            let quantizedSize = (count + 1) / 2
            var quantizedData = [UInt8](repeating: 0, count: quantizedSize)
            var dequantizedData = [Float](repeating: 0, count: count)

            originalData.withUnsafeBufferPointer { inputPtr in
                quantizedData.withUnsafeMutableBufferPointer { quantizedPtr in
                    GEMMOperandPrecision.INT4.quantize(
                        input: inputPtr.baseAddress!,
                        output: UnsafeMutableRawPointer(quantizedPtr.baseAddress!),
                        count: count,
                        parameters: params
                    )
                }
            }

            quantizedData.withUnsafeBufferPointer { quantizedPtr in
                dequantizedData.withUnsafeMutableBufferPointer { dequantizedPtr in
                    GEMMOperandPrecision.INT4.dequantize(
                        input: UnsafeRawPointer(quantizedPtr.baseAddress!),
                        output: dequantizedPtr.baseAddress!,
                        count: count,
                        parameters: params
                    )
                }
            }

            // Check that dequantized values are close to original
            for i in 0..<count {
                let error = abs(dequantizedData[i] - originalData[i])
                let tolerance = params.scale * 2  // Allow for quantization error
                XCTAssertLessThan(error, tolerance,
                    "INT4 quantization error too large at index \(i): \(error) > \(tolerance)")
            }
        }
    }

    func testQuantizedTensorCreation() {
        let testData: [Float] = (0..<100).map { Float($0) * 0.1 - 5.0 }
        let shape = [10, 10]

        // Test INT8 quantized tensor
        let int8Tensor = QuantizedTensor.from(
            device: device,
            floatData: testData,
            shape: shape,
            precision: .INT8
        )

        XCTAssertEqual(int8Tensor.elementCount, 100)
        XCTAssertEqual(int8Tensor.originalShape, shape)
        XCTAssertEqual(int8Tensor.parameters.precision, .INT8)

        // Test round-trip conversion
        let reconstructed = int8Tensor.toFloats()
        XCTAssertEqual(reconstructed.count, testData.count)

        for i in 0..<testData.count {
            let error = abs(reconstructed[i] - testData[i])
            let tolerance = int8Tensor.parameters.scale * 2
            XCTAssertLessThan(error, tolerance,
                "Reconstructed value error too large at index \(i)")
        }
    }

    func testQuantizedAttentionConfiguration() {
        var config = QuantizedAttention.Configuration()
        config.queryPrecision = .FP16
        config.keyPrecision = .INT8
        config.valuePrecision = .INT4

        XCTAssertFalse(config.queryPrecision.requiresQuantizationParameters)
        XCTAssertTrue(config.keyPrecision.requiresQuantizationParameters)
        XCTAssertTrue(config.valuePrecision.requiresQuantizationParameters)

        var baseDescriptor = AttentionDescriptor()
        baseDescriptor.matrixDimensions = (row: 128, column: 128, head: 64)
        baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

        let quantizedDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
            baseDescriptor: baseDescriptor,
            quantizationConfig: config
        )

        let kernelDesc = quantizedDescriptor.kernelDescriptor(type: .forward)

        // Verify that quantized precisions are set correctly
        XCTAssertEqual(kernelDesc.memoryPrecisions[.Q], .FP16)
        XCTAssertEqual(kernelDesc.memoryPrecisions[.K], .INT8)
        XCTAssertEqual(kernelDesc.memoryPrecisions[.V], .INT4)

        // Verify that register precisions are set to FP32 for quantized inputs
        XCTAssertEqual(kernelDesc.registerPrecisions[.K], .FP32)
        XCTAssertEqual(kernelDesc.registerPrecisions[.V], .FP32)
    }

    func testSmallQuantizedAttentionForward() {
        let batchSize = 1
        let sequenceLength = 32
        let headDim = 16

        let totalElements = batchSize * sequenceLength * headDim

        // Generate small test data
        let queryData = (0..<totalElements).map { Float($0) * 0.01 }
        let keyData = (0..<totalElements).map { Float($0 + 1) * 0.01 }
        let valueData = (0..<totalElements).map { Float($0 + 2) * 0.01 }

        let shape = [batchSize, sequenceLength, headDim]

        // Test INT8 configuration
        var config = QuantizedAttention.Configuration()
        config.queryPrecision = .FP16
        config.keyPrecision = .INT8
        config.valuePrecision = .INT8

        let tensors = quantizedAttention.createQuantizedTensors(
            queryData: queryData, keyData: keyData, valueData: valueData,
            queryShape: shape, keyShape: shape, valueShape: shape,
            config: config
        )

        guard let outputBuffer = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            XCTFail("Could not create output buffer")
            return
        }

        var baseDescriptor = AttentionDescriptor()
        baseDescriptor.matrixDimensions = (row: UInt32(sequenceLength), column: UInt32(sequenceLength), head: UInt16(headDim))
        baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

        let descriptor = QuantizedAttention.QuantizedAttentionDescriptor(
            baseDescriptor: baseDescriptor,
            quantizationConfig: config
        )

        // This would normally execute the kernel, but for now we just test creation
        let commandBuffer = quantizedAttention.forward(
            query: tensors.query,
            key: tensors.key,
            value: tensors.value,
            output: outputBuffer,
            descriptor: descriptor
        )

        XCTAssertNotNil(commandBuffer, "Failed to create command buffer")
    }

    func testPerformanceBenchmark() {
        // Run a small benchmark to ensure the API works
        let results = quantizedAttention.benchmark(
            batchSize: 1,
            sequenceLength: 64,  // Small size for test
            headDim: 32,
            iterations: 5
        )

        // Check that we get timing results for each configuration
        XCTAssertNotNil(results["FP16_avg_ms"])
        XCTAssertNotNil(results["INT8_avg_ms"])
        XCTAssertNotNil(results["INT4_avg_ms"])

        // Check that we get GOPS measurements
        XCTAssertNotNil(results["FP16_gops"])
        XCTAssertNotNil(results["INT8_gops"])
        XCTAssertNotNil(results["INT4_gops"])

        print("Benchmark results: \(results)")
    }

    func testMemoryEfficiency() {
        let elementCount = 1024

        // Create test data
        let floatData = (0..<elementCount).map { Float($0) * 0.001 }

        // Test memory usage for different precisions
        let fp32Size = elementCount * MemoryLayout<Float>.size
        let fp16Size = elementCount * MemoryLayout<UInt16>.size

        let int8Tensor = QuantizedTensor.from(
            device: device,
            floatData: floatData,
            shape: [elementCount],
            precision: .INT8
        )
        let int8Size = int8Tensor.data.length

        let int4Tensor = QuantizedTensor.from(
            device: device,
            floatData: floatData,
            shape: [elementCount],
            precision: .INT4
        )
        let int4Size = int4Tensor.data.length

        print("Memory usage comparison:")
        print("FP32: \(fp32Size) bytes")
        print("FP16: \(fp16Size) bytes (\(Float(fp16Size)/Float(fp32Size) * 100)% of FP32)")
        print("INT8: \(int8Size) bytes (\(Float(int8Size)/Float(fp32Size) * 100)% of FP32)")
        print("INT4: \(int4Size) bytes (\(Float(int4Size)/Float(fp32Size) * 100)% of FP32)")

        // Verify expected memory reductions
        XCTAssertEqual(int8Size, elementCount)  // 1 byte per element
        XCTAssertEqual(int4Size, (elementCount + 1) / 2)  // 0.5 bytes per element (packed)

        // Verify significant memory savings
        XCTAssertLessThan(Float(int8Size), Float(fp32Size) * 0.3)  // Less than 30% of FP32
        XCTAssertLessThan(Float(int4Size), Float(fp32Size) * 0.15)  // Less than 15% of FP32
    }
}
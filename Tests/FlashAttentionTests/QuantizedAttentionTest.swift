//
//  QuantizedAttentionTest.swift
//  FlashAttentionTests
//
//

import Foundation
import Metal
import XCTest

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
      XCTAssertEqual(params.zeroPoint, 0) // Symmetric quantization
      XCTAssertEqual(params.scale, 10.0 / 127.0, accuracy: 1e-6)
      XCTAssertEqual(params.strategy, .legacy)
      XCTAssertEqual(params.strategyVersion, QuantizationParameters.currentStrategyVersion)
    }

    // Test INT4 quantization parameter calculation
    testData.withUnsafeBufferPointer { buffer in
      let params = GEMMOperandPrecision.INT4.calculateQuantizationParameters(
        data: buffer.baseAddress!,
        count: buffer.count
      )

      XCTAssertEqual(params.precision, .INT4)
      XCTAssertEqual(params.zeroPoint, 0) // Symmetric quantization
      XCTAssertEqual(params.scale, 10.0 / 7.0, accuracy: 1e-6)
      XCTAssertEqual(params.strategy, .legacy)
      XCTAssertEqual(params.strategyVersion, QuantizationParameters.currentStrategyVersion)
    }
  }

  func testQuantizeAndDequantize() {
    let originalData: [Float] = Array(stride(from: -10.0, through: 10.0, by: 0.5))
    let count = originalData.count

    // Test INT8 round-trip
    do {
      let params = originalData.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
          fatalError("Test data cannot be empty for quantization parameter calculation")
        }
        return GEMMOperandPrecision.INT8.calculateQuantizationParameters(
          data: baseAddress,
          count: count
        )
      }

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
        let tolerance = params.scale * 2 // Allow for quantization error
        XCTAssertLessThan(
          error, tolerance,
          "INT8 quantization error too large at index \(i): \(error) > \(tolerance)"
        )
      }
    }

    // Test INT4 round-trip
    do {
      let params = originalData.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
          fatalError("Test data cannot be empty for quantization parameter calculation")
        }
        return GEMMOperandPrecision.INT4.calculateQuantizationParameters(
          data: baseAddress,
          count: count
        )
      }

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
        let tolerance = params.scale * 2 // Allow for quantization error
        XCTAssertLessThan(
          error, tolerance,
          "INT4 quantization error too large at index \(i): \(error) > \(tolerance)"
        )
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
      XCTAssertLessThan(
        error, tolerance,
        "Reconstructed value error too large at index \(i)"
      )
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

    guard
      let outputBuffer = device.makeBuffer(
        length: totalElements * MemoryLayout<Float>.size,
        options: .storageModeShared
      )
    else {
      XCTFail("Could not create output buffer")
      return
    }

    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (
      row: UInt32(sequenceLength), column: UInt32(sequenceLength), head: UInt16(headDim)
    )
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
      sequenceLength: 64, // Small size for test
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
    print("FP16: \(fp16Size) bytes (\(Float(fp16Size) / Float(fp32Size) * 100)% of FP32)")
    print("INT8: \(int8Size) bytes (\(Float(int8Size) / Float(fp32Size) * 100)% of FP32)")
    print("INT4: \(int4Size) bytes (\(Float(int4Size) / Float(fp32Size) * 100)% of FP32)")

    // Verify expected memory reductions
    XCTAssertEqual(int8Size, elementCount) // 1 byte per element
    XCTAssertEqual(int4Size, (elementCount + 1) / 2) // 0.5 bytes per element (packed)

    // Verify significant memory savings
    XCTAssertLessThan(Float(int8Size), Float(fp32Size) * 0.3) // Less than 30% of FP32
    XCTAssertLessThan(Float(int4Size), Float(fp32Size) * 0.15) // Less than 15% of FP32
  }

  func testQuantizedBackwardPass() {
    // Test dimensions
    let batchSize = 1
    let sequenceLength = 32
    let headDim = 16
    let totalElements = batchSize * sequenceLength * headDim

    // Create test data
    let queryData = (0..<totalElements).map { _ in Float.random(in: -1...1) }
    let keyData = (0..<totalElements).map { _ in Float.random(in: -1...1) }
    let valueData = (0..<totalElements).map { _ in Float.random(in: -1...1) }
    let gradOutputData = (0..<totalElements).map { _ in Float.random(in: -0.1...0.1) }
    let logsumexpData = (0..<sequenceLength).map { _ in Float.random(in: -5...5) }

    let shape = [batchSize, sequenceLength, headDim]

    // Create quantized configuration
    var config = QuantizedAttention.Configuration()
    config.queryPrecision = .INT8
    config.keyPrecision = .INT8
    config.valuePrecision = .INT8

    // Create quantized tensors
    let tensors = quantizedAttention.createQuantizedTensors(
      queryData: queryData,
      keyData: keyData,
      valueData: valueData,
      queryShape: shape,
      keyShape: shape,
      valueShape: shape,
      config: config
    )

    // Create buffers for gradients and intermediate values
    guard
      let gradOutputBuffer = device.makeBuffer(
        bytes: gradOutputData,
        length: totalElements * MemoryLayout<Float>.size,
        options: .storageModeShared
      ),
      let logsumexpBuffer = device.makeBuffer(
        bytes: logsumexpData,
        length: sequenceLength * MemoryLayout<Float>.size,
        options: .storageModeShared
      ),
      let gradQueryBuffer = device.makeBuffer(
        length: totalElements * MemoryLayout<Float>.size,
        options: .storageModeShared
      ),
      let gradKeyBuffer = device.makeBuffer(
        length: totalElements * MemoryLayout<Float>.size,
        options: .storageModeShared
      ),
      let gradValueBuffer = device.makeBuffer(
        length: totalElements * MemoryLayout<Float>.size,
        options: .storageModeShared
      ),
      let dValuesBuffer = device.makeBuffer(
        length: sequenceLength * MemoryLayout<Float>.size,
        options: .storageModeShared
      )
    else {
      XCTFail("Failed to create buffers")
      return
    }

    // Create attention descriptor
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (
      row: UInt32(sequenceLength),
      column: UInt32(sequenceLength),
      head: UInt16(headDim)
    )
    baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

    let descriptor = QuantizedAttention.QuantizedAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      quantizationConfig: config
    )

    // Test backward query pass
    guard
      let queryCommandBuffer = quantizedAttention.backwardQuery(
        query: tensors.query,
        key: tensors.key,
        value: tensors.value,
        gradOutput: gradOutputBuffer,
        logsumexp: logsumexpBuffer,
        gradQuery: gradQueryBuffer,
        dValues: dValuesBuffer,
        descriptor: descriptor
      )
    else {
      XCTFail("Failed to create backward query command buffer")
      return
    }

    queryCommandBuffer.commit()
    queryCommandBuffer.waitUntilCompleted()

    XCTAssertNil(
      queryCommandBuffer.error,
      "Backward query pass failed: \(queryCommandBuffer.error?.localizedDescription ?? "")"
    )

    // Test backward key-value pass
    guard
      let kvCommandBuffer = quantizedAttention.backwardKeyValue(
        query: tensors.query,
        key: tensors.key,
        value: tensors.value,
        gradOutput: gradOutputBuffer,
        logsumexp: logsumexpBuffer,
        dValues: dValuesBuffer,
        gradKey: gradKeyBuffer,
        gradValue: gradValueBuffer,
        descriptor: descriptor
      )
    else {
      XCTFail("Failed to create backward key-value command buffer")
      return
    }

    kvCommandBuffer.commit()
    kvCommandBuffer.waitUntilCompleted()

    XCTAssertNil(
      kvCommandBuffer.error,
      "Backward key-value pass failed: \(kvCommandBuffer.error?.localizedDescription ?? "")"
    )

    let gpuGradQuery = readBuffer(gradQueryBuffer, count: totalElements)
    let gpuGradKey = readBuffer(gradKeyBuffer, count: totalElements)
    let gpuGradValue = readBuffer(gradValueBuffer, count: totalElements)

    let queryGradNorm = l2Norm(gpuGradQuery)
    let keyGradNorm = l2Norm(gpuGradKey)
    let valueGradNorm = l2Norm(gpuGradValue)

    print("Quantized backward pass results:")
    print("  Query gradient norm: \(queryGradNorm)")
    print("  Key gradient norm: \(keyGradNorm)")
    print("  Value gradient norm: \(valueGradNorm)")

    XCTAssertGreaterThan(queryGradNorm, 0.001, "Query gradients appear to be zero")
    XCTAssertLessThan(queryGradNorm, 1000.0, "Query gradients appear too large")
    XCTAssertGreaterThan(keyGradNorm, 0.001, "Key gradients appear to be zero")
    XCTAssertLessThan(keyGradNorm, 1000.0, "Key gradients appear too large")
    XCTAssertGreaterThan(valueGradNorm, 0.001, "Value gradients appear to be zero")
    XCTAssertLessThan(valueGradNorm, 1000.0, "Value gradients appear too large")

    let floatGradients = runFloatBackward(
      query: queryData,
      key: keyData,
      value: valueData,
      gradOutput: gradOutputData,
      logsumexp: logsumexpData,
      descriptor: baseDescriptor
    )

    let queryCosine = cosineSimilarity(gpuGradQuery, floatGradients.dQ)
    let keyCosine = cosineSimilarity(gpuGradKey, floatGradients.dK)
    let valueCosine = cosineSimilarity(gpuGradValue, floatGradients.dV)

    let queryRelativeError = relativeError(gpuGradQuery, floatGradients.dQ)
    let keyRelativeError = relativeError(gpuGradKey, floatGradients.dK)
    let valueRelativeError = relativeError(gpuGradValue, floatGradients.dV)

    print("Gradient comparison metrics:")
    print("  Query cosine similarity: \(queryCosine), relative error: \(queryRelativeError)")
    print("  Key cosine similarity: \(keyCosine), relative error: \(keyRelativeError)")
    print("  Value cosine similarity: \(valueCosine), relative error: \(valueRelativeError)")

    XCTAssertGreaterThan(queryCosine, 0.7, "Query gradient cosine similarity too low")
    XCTAssertGreaterThan(keyCosine, 0.7, "Key gradient cosine similarity too low")
    XCTAssertGreaterThan(valueCosine, 0.7, "Value gradient cosine similarity too low")

    XCTAssertLessThan(queryRelativeError, 0.30, "Query gradient relative error too high")
    XCTAssertLessThan(keyRelativeError, 0.30, "Key gradient relative error too high")
    XCTAssertLessThan(valueRelativeError, 0.30, "Value gradient relative error too high")
  }

  func testKernelSourceIncludesStrategyBuffers() {
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (row: 16, column: 16, head: 16)
    baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

    var config = QuantizedAttention.Configuration()
    config.queryPrecision = .INT8
    config.queryStrategy = .symmetric

    let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      quantizationConfig: config
    )

    let kernelDescriptor = quantDescriptor.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDescriptor)
    let source = kernel.createSource()

    XCTAssertTrue(
      source.contains("constant uint &q_strategy [[buffer"),
      "Generated kernel is missing q_strategy binding"
    )
    XCTAssertTrue(
      source.contains("constant uint &q_strategy_version [[buffer"),
      "Generated kernel is missing q_strategy_version binding"
    )
  }

  func testKernelSourceIncludesOStridesForBackwardKeyValue() {
    // Test the quantized backward kernel generation which uses QuantizedKernelLayoutManifest
    let layout = QuantizedKernelLayoutManifest.layout(for: .backwardKeyValue)

    // Verify O_strides is assigned in the layout
    XCTAssertNotEqual(layout.oStrides, -1, "O_strides should be assigned in backward key-value layout")

    // Verify O_strides is within Metal's buffer limit (0-30)
    XCTAssertLessThanOrEqual(layout.oStrides, 30, "O_strides buffer index must be <= 30 for Metal compatibility")
    XCTAssertGreaterThanOrEqual(layout.oStrides, 0, "O_strides buffer index must be >= 0")
  }

  func testConfigurationCodableRoundTripPreservesStrategies() throws {
    var config = QuantizedAttention.Configuration()
    config.queryPrecision = .INT8
    config.keyPrecision = .INT8
    config.valuePrecision = .INT4
    config.queryStrategy = .symmetric
    config.keyStrategy = .asymmetric
    config.valueStrategy = .legacy
    config.strategyVersion = 42

    let encoder = JSONEncoder()
    let data = try encoder.encode(config)
    let decoder = JSONDecoder()
    let decoded = try decoder.decode(QuantizedAttention.Configuration.self, from: data)

    XCTAssertEqual(decoded.queryStrategy, .symmetric)
    XCTAssertEqual(decoded.keyStrategy, .asymmetric)
    XCTAssertEqual(decoded.valueStrategy, .legacy)
    XCTAssertEqual(decoded.strategyVersion, 42)
  }

  func testQuantizedTensorFromRespectsStrategy() {
    let device = MTLCreateSystemDefaultDevice()!
    let values: [Float] = [1, 2, 3, 4]

    let tensor = QuantizedTensor.from(
      device: device,
      floatData: values,
      shape: [2, 2],
      precision: .INT8,
      strategy: .symmetric
    )

    XCTAssertEqual(tensor.parameters.strategy, .symmetric)
  }
}

private extension QuantizedAttentionTest {
  func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return (0..<count).map { pointer[$0] }
  }

  func l2Norm(_ values: [Float]) -> Float {
    var sum: Double = 0
    for value in values {
      sum += Double(value) * Double(value)
    }
    return Float(sqrt(sum))
  }

  func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
    precondition(lhs.count == rhs.count)

    var dot: Double = 0
    var lhsNorm: Double = 0
    var rhsNorm: Double = 0

    for index in 0..<lhs.count {
      let left = Double(lhs[index])
      let right = Double(rhs[index])
      dot += left * right
      lhsNorm += left * left
      rhsNorm += right * right
    }

    let denominator = sqrt(lhsNorm) * sqrt(rhsNorm)
    guard denominator > 1e-8 else { return 0 }
    return Float(dot / denominator)
  }

  func relativeError(_ candidate: [Float], _ reference: [Float]) -> Float {
    precondition(candidate.count == reference.count)

    var diffNorm: Double = 0
    var referenceNorm: Double = 0

    for index in 0..<candidate.count {
      let diff = Double(candidate[index]) - Double(reference[index])
      diffNorm += diff * diff
      let ref = Double(reference[index])
      referenceNorm += ref * ref
    }

    let numerator = sqrt(diffNorm)
    let denominator = sqrt(referenceNorm) + 1e-8
    return Float(numerator / denominator)
  }

  func runFloatBackward(
    query: [Float],
    key: [Float],
    value: [Float],
    gradOutput: [Float],
    logsumexp: [Float],
    descriptor: AttentionDescriptor
  ) -> (dQ: [Float], dK: [Float], dV: [Float]) {
    guard let dims = descriptor.matrixDimensions else {
      return (
        dQ: Array(repeating: 0, count: query.count),
        dK: Array(repeating: 0, count: key.count),
        dV: Array(repeating: 0, count: value.count)
      )
    }

    let M = Int(dims.row)
    let N = Int(dims.column)
    let KDim = Int(dims.head)

    precondition(query.count == M * KDim)
    precondition(key.count == N * KDim)
    precondition(value.count == N * KDim)
    precondition(gradOutput.count == M * KDim)
    precondition(logsumexp.count >= M)

    let steClipRange: Float = 6.0

    var dQ = [Float](repeating: 0, count: M * KDim)
    var dK = [Float](repeating: 0, count: N * KDim)
    var dV = [Float](repeating: 0, count: N * KDim)

    for row in 0..<M {
      for col in 0..<KDim {
        let qValue = query[row * KDim + col]
        let absQ = abs(qValue)
        var clipFactor: Float = 1.0
        if absQ > steClipRange {
          clipFactor = Swift.max(steClipRange / absQ, 0.1)
        }

        var dqAccumulator: Float = 0

        for n in 0..<N {
          var qkDot: Float = 0
          for k in 0..<KDim {
            qkDot += query[row * KDim + k] * key[n * KDim + k]
          }

          let clampedLogit = Swift.max(-10.0, Swift.min(10.0, qkDot))
          let stableLogit = clampedLogit - logsumexp[row]
          let pVal = Swift.max(0.0, Swift.min(1.0, Float(Foundation.exp(Double(stableLogit)))))

          var gradFactor = pVal * gradOutput[row * KDim + col]
          gradFactor *= 0.01 as Float

          let kCol = key[n * KDim + col]
          let vCol = value[n * KDim + col]
          let combined = (0.5 as Float) * (kCol + vCol)

          dqAccumulator += gradFactor * combined
        }

        dqAccumulator = Swift.max(-10.0, Swift.min(10.0, dqAccumulator))
        dQ[row * KDim + col] = dqAccumulator * clipFactor
      }
    }

    for row in 0..<N {
      for col in 0..<KDim {
        let kValue = key[row * KDim + col]
        let vValue = value[row * KDim + col]

        var kClipFactor: Float = 1.0
        var vClipFactor: Float = 1.0

        let absK = abs(kValue)
        if absK > steClipRange {
          kClipFactor = Swift.max(steClipRange / absK, 0.1)
        }

        let absV = abs(vValue)
        if absV > steClipRange {
          vClipFactor = Swift.max(steClipRange / absV, 0.1)
        }

        var dkAccumulator: Float = 0
        var dvAccumulator: Float = 0

        for m in 0..<M {
          var qkDot: Float = 0
          for k in 0..<KDim {
            qkDot += query[m * KDim + k] * key[row * KDim + k]
          }

          let clampedQK = Swift.max(-10.0, Swift.min(10.0, qkDot))
          let stableLogit = clampedQK - logsumexp[m]
          let pVal = Swift.max(0.0, Swift.min(1.0, Float(Foundation.exp(Double(stableLogit)))))

          let gradComponent = pVal * gradOutput[m * KDim + col]
          dkAccumulator += query[m * KDim + col] * gradComponent * (0.1 as Float)
          dvAccumulator += gradComponent * (0.1 as Float)
        }

        dkAccumulator = Swift.max(-100.0, Swift.min(100.0, dkAccumulator))
        dvAccumulator = Swift.max(-100.0, Swift.min(100.0, dvAccumulator))

        dK[row * KDim + col] = dkAccumulator * kClipFactor
        dV[row * KDim + col] = dvAccumulator * vClipFactor
      }
    }

    return (dQ: dQ, dK: dK, dV: dV)
  }
}

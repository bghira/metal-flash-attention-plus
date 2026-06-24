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

  /// Real correctness check for the quantized forward path.
  ///
  /// Unlike `testSmallQuantizedAttentionForward` (which only checks that a
  /// command buffer is created), this commits the kernel, reads back the
  /// output, and compares it against a CPU softmax(QKᵀ/√d)V reference computed
  /// from the original (pre-quantization) float data. The dispatch site must
  /// stay in lockstep with `AttentionKernel.createBufferBindings()` for this to
  /// pass; a desync shows up immediately as NaN / garbage output.
  func testQuantizedForwardCorrectness() {
    let sequenceLength = 32
    let headDim = 16
    let totalElements = sequenceLength * headDim

    // Deterministic pseudo-random data.
    var seed: UInt64 = 0x5EED_5EED
    func nextRandom() -> Float {
      seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max)
    }
    let queryData = (0..<totalElements).map { _ in nextRandom() * 2 - 1 }
    let keyData = (0..<totalElements).map { _ in nextRandom() * 2 - 1 }
    let valueData = (0..<totalElements).map { _ in nextRandom() * 2 - 1 }

    // CPU reference: attention(Q, K, V) using the original float data.
    let reference = Self.cpuReferenceAttention(
      query: queryData, key: keyData, value: valueData,
      rows: sequenceLength, cols: sequenceLength, headDim: headDim
    )

    let shape = [sequenceLength, headDim]

    // FP16 baseline first — exercises the same kernel without quantization
    // dequantization, isolating dispatch correctness from quantization error.
    do {
      var fp16Config = QuantizedAttention.Configuration()
      fp16Config.queryPrecision = .FP16
      fp16Config.keyPrecision = .FP16
      fp16Config.valuePrecision = .FP16

      let tensors = quantizedAttention.createQuantizedTensors(
        queryData: queryData, keyData: keyData, valueData: valueData,
        queryShape: shape, keyShape: shape, valueShape: shape,
        config: fp16Config
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
        baseDescriptor: baseDescriptor, quantizationConfig: fp16Config
      )

      guard
        let commandBuffer = quantizedAttention.forward(
          query: tensors.query, key: tensors.key, value: tensors.value,
          output: outputBuffer, descriptor: descriptor
        )
      else {
        XCTFail("FP16 forward returned nil command buffer")
        return
      }
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      XCTAssertNil(
        commandBuffer.error,
        "FP16 forward failed: \(commandBuffer.error?.localizedDescription ?? "")"
      )

      let gpuOutput = readBuffer(outputBuffer, count: totalElements)
      let relErr = relativeError(gpuOutput, reference)
      let nanCount = gpuOutput.filter { $0.isNaN || $0.isInfinite }.count
      print("FP16 forward: relativeError=\(relErr), NaN/Inf=\(nanCount)/\(totalElements)")
      XCTAssertLessThan(nanCount, 1, "FP16 forward produced NaN/Inf output")
      XCTAssertLessThan(relErr, 0.05, "FP16 forward relative error too high (dispatch desync?)")
    }

    // INT8 quantized path — wider tolerance to allow for quantization error.
    do {
      var int8Config = QuantizedAttention.Configuration()
      int8Config.queryPrecision = .INT8
      int8Config.keyPrecision = .INT8
      int8Config.valuePrecision = .INT8

      let tensors = quantizedAttention.createQuantizedTensors(
        queryData: queryData, keyData: keyData, valueData: valueData,
        queryShape: shape, keyShape: shape, valueShape: shape,
        config: int8Config
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
        baseDescriptor: baseDescriptor, quantizationConfig: int8Config
      )

      guard
        let commandBuffer = quantizedAttention.forward(
          query: tensors.query, key: tensors.key, value: tensors.value,
          output: outputBuffer, descriptor: descriptor
        )
      else {
        XCTFail("INT8 forward returned nil command buffer")
        return
      }
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      XCTAssertNil(
        commandBuffer.error,
        "INT8 forward failed: \(commandBuffer.error?.localizedDescription ?? "")"
      )

      let gpuOutput = readBuffer(outputBuffer, count: totalElements)
      let relErr = relativeError(gpuOutput, reference)
      let nanCount = gpuOutput.filter { $0.isNaN || $0.isInfinite }.count
      print("INT8 forward: relativeError=\(relErr), NaN/Inf=\(nanCount)/\(totalElements)")
      XCTAssertLessThan(nanCount, 1, "INT8 forward produced NaN/Inf output")
      XCTAssertLessThan(relErr, 0.25, "INT8 forward relative error too high")
    }
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

  /// Validates that `QuantizedTensor.from(mode: .blockwise)` actually quantizes
  /// per-block and round-trips through `toFloats()`. Previously the factory
  /// discarded the per-block scales, quantized with a single tensor-wise scale,
  /// and left `blockScales` nil — so blockwise silently behaved like tensor-wise.
  func testBlockwiseQuantizationRoundTrip() {
    let rows = 16
    let cols = 32
    let blockSize = 8
    let totalElements = rows * cols

    // Skewed data so different blocks have genuinely different scales.
    let data = (0..<totalElements).map { i -> Float in
      let blockRow = (i / cols) / blockSize
      let blockCol = (i % cols) / blockSize
      let magnitude = Float((blockRow + 1) * (blockCol + 1))
      return (Float(i % 7) - 3) * magnitude
    }

    let tensor = QuantizedTensor.from(
      device: device,
      floatData: data,
      shape: [rows, cols],
      precision: .INT8,
      mode: .blockwise(blockSizeK: blockSize)
    )

    XCTAssertNotNil(tensor.blockScales, "blockScales must be populated for blockwise tensors")
    XCTAssertNotNil(tensor.blockZeroPoints)
    XCTAssertEqual(tensor.blockSizeK, blockSize)

    let expectedNumBlocks = ((rows + blockSize - 1) / blockSize) *
      ((cols + blockSize - 1) / blockSize)
    XCTAssertEqual(
      tensor.blockScales!.length / MemoryLayout<Float>.size,
      expectedNumBlocks,
      "blockScales count must match the 2D block grid"
    )

    let reconstructed = tensor.toFloats()
    XCTAssertEqual(reconstructed.count, totalElements)

    // Each element's error must be bounded by its own block's scale (≈1 LSB),
    // not the global max scale — that's the whole point of blockwise.
    let allScales = Array(UnsafeBufferPointer(
      start: tensor.blockScales!.contents().bindMemory(
        to: Float.self, capacity: expectedNumBlocks
      ),
      count: expectedNumBlocks
    ))
    let numBlocksCol = (cols + blockSize - 1) / blockSize

    var maxRelativeError: Float = 0
    for i in 0..<totalElements {
      let r = i / cols
      let c = i % cols
      let blockIdx = (r / blockSize) * numBlocksCol + (c / blockSize)
      let blockScale = allScales[blockIdx]
      let error = abs(reconstructed[i] - data[i])
      // Allow ±2 LSB of quantization slack per element.
      XCTAssertLessThanOrEqual(
        error, blockScale * 2.01,
        "Blockwise round-trip error \(error) exceeds block scale \(blockScale) at index \(i)"
      )
      if blockScale > 0 {
        maxRelativeError = max(maxRelativeError, error / blockScale)
      }
    }
    print("Blockwise INT8 round-trip: maxRelativeError( / blockScale) = \(maxRelativeError)")
  }

  /// End-to-end blockwise attention forward: FP16 Q with blockwise-INT8 K and V
  /// (the common LLM-inference configuration). Compares the GPU output against
  /// the CPU softmax(QKᵀ/√d)V reference computed from the original float data.
  /// Uses headDim=32, blockSize=8 → a 4×4 block grid, so the 2D block indexing
  /// is genuinely exercised (not a single-block degenerate case).
  func testBlockwiseAttentionForward() {
    let sequenceLength = 32
    let headDim = 32
    let blockSize = 8
    let totalElements = sequenceLength * headDim

    // Skewed data so each block has a distinct scale.
    let queryData = (0..<totalElements).map { i -> Float in
      Float(i % 11) * 0.05 - 0.25
    }
    let keyData = (0..<totalElements).map { i -> Float in
      let br = (i / headDim) / blockSize
      let bc = (i % headDim) / blockSize
      return (Float(i % 7) - 3) * Float((br + 1) * (bc + 1)) * 0.1
    }
    let valueData = (0..<totalElements).map { i -> Float in
      let br = (i / headDim) / blockSize
      let bc = (i % headDim) / blockSize
      return (Float(i % 5) - 2) * Float((br + 1) * (bc + 1)) * 0.1
    }

    let reference = Self.cpuReferenceAttention(
      query: queryData, key: keyData, value: valueData,
      rows: sequenceLength, cols: sequenceLength, headDim: headDim
    )

    let shape = [sequenceLength, headDim]
    let queryTensor = QuantizedTensor.from(
      device: device, floatData: queryData, shape: shape, precision: .FP16
    )
    let keyTensor = QuantizedTensor.from(
      device: device, floatData: keyData, shape: shape, precision: .INT8,
      mode: .blockwise(blockSizeK: blockSize)
    )
    let valueTensor = QuantizedTensor.from(
      device: device, floatData: valueData, shape: shape, precision: .INT8,
      mode: .blockwise(blockSizeK: blockSize)
    )

    XCTAssertNotNil(keyTensor.blockScales)
    XCTAssertNotNil(valueTensor.blockScales)

    guard
      let outputBuffer = device.makeBuffer(
        length: totalElements * MemoryLayout<Float>.size, options: .storageModeShared
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

    var config = QuantizedAttention.Configuration()
    config.queryPrecision = .FP16
    config.keyPrecision = .INT8
    config.valuePrecision = .INT8
    let descriptor = QuantizedAttention.QuantizedAttentionDescriptor(
      baseDescriptor: baseDescriptor, quantizationConfig: config
    )

    guard
      let commandBuffer = quantizedAttention.forward(
        query: queryTensor, key: keyTensor, value: valueTensor,
        output: outputBuffer, descriptor: descriptor
      )
    else {
      XCTFail("Blockwise forward returned nil command buffer")
      return
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertNil(
      commandBuffer.error,
      "Blockwise forward failed: \(commandBuffer.error?.localizedDescription ?? "")"
    )

    let gpuOutput = readBuffer(outputBuffer, count: totalElements)
    let relErr = relativeError(gpuOutput, reference)
    let nanCount = gpuOutput.filter { $0.isNaN || $0.isInfinite }.count
    print(
      "Blockwise FP16-Q/INT8-K,V forward: relativeError=\(relErr), NaN/Inf=\(nanCount)/\(totalElements)"
    )

    XCTAssertEqual(nanCount, 0, "Blockwise forward produced NaN/Inf output")
    // Blockwise INT8 should be markedly more accurate than a single tensor-wise
    // scale would be for this skewed data; 0.15 is a comfortable ceiling.
    XCTAssertLessThan(
      relErr,
      0.15,
      "Blockwise forward relative error too high (2D block indexing wrong?)"
    )
  }

  /// Real end-to-end backward correctness check.
  ///
  /// Runs the actual forward→backward graph through the GPU (now dispatched on
  /// the proven core kernel) and compares dQ/dK/dV against a proper CPU
  /// flash-attention backward reference. The previous version of this test
  /// compared the GPU against a CPU routine that mirrored the GPU's own
  /// placeholder math — a tautology that passed while the gradients were wrong.
  func testQuantizedBackwardPass() {
    let sequenceLength = 32
    let headDim = 16
    let totalElements = sequenceLength * headDim

    var seed: UInt64 = 0xBACC0DE
    func nextRandom() -> Float {
      seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max)
    }
    let queryData = (0..<totalElements).map { _ in nextRandom() * 2 - 1 }
    let keyData = (0..<totalElements).map { _ in nextRandom() * 2 - 1 }
    let valueData = (0..<totalElements).map { _ in nextRandom() * 2 - 1 }
    let gradOutputData = (0..<totalElements).map { _ in nextRandom() * 0.2 - 0.1 }

    let reference = Self.cpuReferenceBackward(
      query: queryData, key: keyData, value: valueData, gradOutput: gradOutputData,
      rows: sequenceLength, cols: sequenceLength, headDim: headDim
    )

    let shape = [sequenceLength, headDim]

    func runBackward(precision: GEMMOperandPrecision, tolerance: Float, label: String) {
      var config = QuantizedAttention.Configuration()
      config.queryPrecision = precision
      config.keyPrecision = precision
      config.valuePrecision = precision

      let tensors = quantizedAttention.createQuantizedTensors(
        queryData: queryData, keyData: keyData, valueData: valueData,
        queryShape: shape, keyShape: shape, valueShape: shape, config: config
      )

      guard
        let outputBuffer = device.makeBuffer(
          bytes: reference.output,
          length: totalElements * MemoryLayout<Float>.size, options: .storageModeShared
        ),
        let logsumexpBuffer = device.makeBuffer(
          // The kernel computes logsumexp in log base 2 (fast::exp2), so scale
          // the natural-log reference by log2(e) before passing it in.
          bytes: reference.logsumexp.map { $0 * 1.4426950408889634 },
          length: sequenceLength * MemoryLayout<Float>.size, options: .storageModeShared
        ),
        let gradOutputBuffer = device.makeBuffer(
          bytes: gradOutputData,
          length: totalElements * MemoryLayout<Float>.size, options: .storageModeShared
        ),
        let gradQueryBuffer = device.makeBuffer(
          length: totalElements * MemoryLayout<Float>.size, options: .storageModeShared
        ),
        let gradKeyBuffer = device.makeBuffer(
          length: totalElements * MemoryLayout<Float>.size, options: .storageModeShared
        ),
        let gradValueBuffer = device.makeBuffer(
          length: totalElements * MemoryLayout<Float>.size, options: .storageModeShared
        ),
        let dValuesBuffer = device.makeBuffer(
          length: sequenceLength * MemoryLayout<Float>.size, options: .storageModeShared
        )
      else {
        XCTFail("Failed to create buffers")
        return
      }

      var baseDescriptor = AttentionDescriptor()
      baseDescriptor.matrixDimensions = (
        row: UInt32(sequenceLength), column: UInt32(sequenceLength), head: UInt16(headDim)
      )
      baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)
      let descriptor = QuantizedAttention.QuantizedAttentionDescriptor(
        baseDescriptor: baseDescriptor, quantizationConfig: config
      )

      guard
        let queryCB = quantizedAttention.backwardQuery(
          query: tensors.query, key: tensors.key, value: tensors.value,
          output: outputBuffer, gradOutput: gradOutputBuffer, logsumexp: logsumexpBuffer,
          gradQuery: gradQueryBuffer, dValues: dValuesBuffer, descriptor: descriptor
        )
      else {
        XCTFail("\(label): backwardQuery returned nil")
        return
      }
      queryCB.commit()
      queryCB.waitUntilCompleted()
      XCTAssertNil(
        queryCB.error,
        "\(label): backwardQuery failed: \(queryCB.error?.localizedDescription ?? "")"
      )

      guard
        let kvCB = quantizedAttention.backwardKeyValue(
          query: tensors.query, key: tensors.key, value: tensors.value,
          gradOutput: gradOutputBuffer, logsumexp: logsumexpBuffer, dValues: dValuesBuffer,
          gradKey: gradKeyBuffer, gradValue: gradValueBuffer, descriptor: descriptor
        )
      else {
        XCTFail("\(label): backwardKeyValue returned nil")
        return
      }
      kvCB.commit()
      kvCB.waitUntilCompleted()
      XCTAssertNil(
        kvCB.error,
        "\(label): backwardKeyValue failed: \(kvCB.error?.localizedDescription ?? "")"
      )

      let dQ = readBuffer(gradQueryBuffer, count: totalElements)
      let dK = readBuffer(gradKeyBuffer, count: totalElements)
      let dV = readBuffer(gradValueBuffer, count: totalElements)

      let dQerr = relativeError(dQ, reference.dQ)
      let dKerr = relativeError(dK, reference.dK)
      let dVerr = relativeError(dV, reference.dV)
      let nanCount = dQ.filter { $0.isNaN || $0.isInfinite }.count
        + dK.filter { $0.isNaN || $0.isInfinite }.count
        + dV.filter { $0.isNaN || $0.isInfinite }.count
      print(
        "\(label) backward: dQ_err=\(dQerr), dK_err=\(dKerr), dV_err=\(dVerr), NaN/Inf=\(nanCount)"
      )

      XCTAssertEqual(nanCount, 0, "\(label): backward produced NaN/Inf gradients")
      XCTAssertLessThan(dQerr, tolerance, "\(label): dQ relative error too high")
      XCTAssertLessThan(dKerr, tolerance, "\(label): dK relative error too high")
      XCTAssertLessThan(dVerr, tolerance, "\(label): dV relative error too high")
    }

    runBackward(precision: .FP16, tolerance: 0.05, label: "FP16")
    runBackward(precision: .INT8, tolerance: 0.25, label: "INT8")
  }

  func testKernelSourceIncludesQuantizationBuffers() {
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

    // The dequantizing load path consumes scale and zero_point; block_scales
    // is emitted (null unless blockwise) so the inner loop can override per block.
    XCTAssertTrue(
      source.contains("constant float &q_scale [[buffer"),
      "Generated kernel is missing q_scale binding"
    )
    XCTAssertTrue(
      source.contains("constant int32_t &q_zero_point [[buffer"),
      "Generated kernel is missing q_zero_point binding"
    )
    XCTAssertTrue(
      source.contains("device const float* q_block_scales [[buffer"),
      "Generated kernel is missing q_block_scales binding"
    )
  }

  func testKernelSourceIncludesOStridesForBackwardKeyValue() {
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.matrixDimensions = (row: 16, column: 16, head: 16)
    baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

    var config = QuantizedAttention.Configuration()
    config.queryPrecision = .INT8
    config.keyPrecision = .INT8
    config.valuePrecision = .INT8

    let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      quantizationConfig: config
    )

    let kernelDescriptor = quantDescriptor.kernelDescriptor(type: .backwardKeyValue)
    let kernel = AttentionKernel(descriptor: kernelDescriptor)
    let source = kernel.createSource()

    XCTAssertTrue(
      source.contains("constant int64_t* O_strides [[buffer"),
      "Generated kernel should always declare O_strides"
    )
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
  /// Naive CPU reference: O = softmax(Q·Kᵀ / √d) · V, computed in Float.
  /// Layout is row-major [seq, headDim] for Q/K/V/O.
  static func cpuReferenceAttention(
    query: [Float], key: [Float], value: [Float],
    rows: Int, cols: Int, headDim: Int
  )
    -> [Float]
  {
    precondition(query.count == rows * headDim)
    precondition(key.count == cols * headDim)
    precondition(value.count == cols * headDim)

    let scale = 1 / Float(Float(headDim).squareRoot())
    var output = [Float](repeating: 0, count: rows * headDim)

    for i in 0..<rows {
      var maxScore = -Float.greatestFiniteMagnitude
      var scores = [Float](repeating: 0, count: cols)
      for j in 0..<cols {
        var dot: Float = 0
        for d in 0..<headDim {
          dot += query[i * headDim + d] * key[j * headDim + d]
        }
        scores[j] = dot * scale
        if scores[j] > maxScore { maxScore = scores[j] }
      }

      var sumExp: Float = 0
      for j in 0..<cols {
        scores[j] = Foundation.exp(scores[j] - maxScore)
        sumExp += scores[j]
      }

      for d in 0..<headDim {
        var acc: Float = 0
        for j in 0..<cols {
          acc += (scores[j] / sumExp) * value[j * headDim + d]
        }
        output[i * headDim + d] = acc
      }
    }
    return output
  }

  /// CPU flash-attention backward reference, matching the kernel's math:
  ///   S = Q·Kᵀ·scale,  P = softmax(S),  O = P·V,  L = logsumexp(S)
  ///   D_i = row_d( dO ⊙ O )
  ///   dP = dO·Vᵀ ,  dS = P ⊙ (dP − D) · scale   (= dL/d(QKᵀ))
  ///   dQ = dS·K ,  dK = dSᵀ·Q ,  dV = Pᵀ·dO
  static func cpuReferenceBackward(
    query: [Float], key: [Float], value: [Float], gradOutput: [Float],
    rows: Int, cols: Int, headDim: Int
  )
    -> (logsumexp: [Float], output: [Float], dQ: [Float], dK: [Float], dV: [Float])
  {
    precondition(query.count == rows * headDim)
    precondition(key.count == cols * headDim)
    precondition(value.count == cols * headDim)
    precondition(gradOutput.count == rows * headDim)

    let scale = 1 / Float(Float(headDim).squareRoot())
    var logsumexp = [Float](repeating: 0, count: rows)
    var output = [Float](repeating: 0, count: rows * headDim)
    var probabilities = [[Float]](repeating: [Float](repeating: 0, count: cols), count: rows)

    // Forward pass: P, L, O.
    for i in 0..<rows {
      var maxScore = -Float.greatestFiniteMagnitude
      var scores = [Float](repeating: 0, count: cols)
      for j in 0..<cols {
        var dot: Float = 0
        for d in 0..<headDim {
          dot += query[i * headDim + d] * key[j * headDim + d]
        }
        scores[j] = dot * scale
        if scores[j] > maxScore { maxScore = scores[j] }
      }
      var sumExp: Float = 0
      for j in 0..<cols {
        scores[j] = Foundation.exp(scores[j] - maxScore)
        sumExp += scores[j]
      }
      logsumexp[i] = maxScore + Foundation.log(sumExp)
      for j in 0..<cols {
        probabilities[i][j] = scores[j] / sumExp
      }
      for d in 0..<headDim {
        var acc: Float = 0
        for j in 0..<cols {
          acc += probabilities[i][j] * value[j * headDim + d]
        }
        output[i * headDim + d] = acc
      }
    }

    var dQ = [Float](repeating: 0, count: rows * headDim)
    var dK = [Float](repeating: 0, count: cols * headDim)
    var dV = [Float](repeating: 0, count: cols * headDim)

    for i in 0..<rows {
      // D_i = sum_d dO[i,d] * O[i,d]
      var dRow: Float = 0
      for d in 0..<headDim {
        dRow += gradOutput[i * headDim + d] * output[i * headDim + d]
      }
      for j in 0..<cols {
        // dP_ij = sum_d dO[i,d] * V[j,d]
        var dp: Float = 0
        for d in 0..<headDim {
          dp += gradOutput[i * headDim + d] * value[j * headDim + d]
        }
        let dS = probabilities[i][j] * (dp - dRow) * scale
        for d in 0..<headDim {
          dQ[i * headDim + d] += dS * key[j * headDim + d]
          dK[j * headDim + d] += dS * query[i * headDim + d]
          dV[j * headDim + d] += probabilities[i][j] * gradOutput[i * headDim + d]
        }
      }
    }

    return (logsumexp, output, dQ, dK, dV)
  }

  func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return (0..<count).map { pointer[$0] }
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
}

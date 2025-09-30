//
//  QuantizedBackwardTest.swift
//
//  Tests for quantized backward pass with strategy field encoding
//

import Metal
import XCTest
@testable import FlashAttention

final class QuantizedBackwardTest: XCTestCase {
  var device: MTLDevice!
  var quantizedAttention: QuantizedAttention!

  override func setUp() {
    super.setUp()
    device = MTLCreateSystemDefaultDevice()
    XCTAssertNotNil(device, "Metal device should be available")
    quantizedAttention = QuantizedAttention(device: device)
  }

  override func tearDown() {
    // No need to dispose explicitly
    super.tearDown()
  }

  func testBackwardQueryStrategyEncoding() throws {
    // Test dimensions
    let seqLength: UInt32 = 64
    let headDim: UInt16 = 32
    let elementCount = Int(seqLength * seqLength)
    let tensorSize = elementCount * Int(headDim)

    // Create test tensors with different quantization strategies
    let symmetricParams = QuantizationParameters(
      scale: 0.0625,
      zeroPoint: 0,
      precision: .INT8,
      mode: .tensorWise,
      strategy: .symmetric,
      strategyVersion: 1
    )

    let asymmetricParams = QuantizationParameters(
      scale: 0.0625,
      zeroPoint: 128,
      precision: .INT8,
      mode: .tensorWise,
      strategy: .asymmetric,
      strategyVersion: 1
    )

    // Create quantized tensors
    let queryBuffer = device.makeBuffer(length: tensorSize, options: .storageModeShared)!
    let keyBuffer = device.makeBuffer(length: tensorSize, options: .storageModeShared)!
    let valueBuffer = device.makeBuffer(length: tensorSize, options: .storageModeShared)!

    let query = QuantizedTensor(
      device: device,
      data: queryBuffer,
      parameters: symmetricParams,
      elementCount: tensorSize,
      shape: [Int(seqLength), Int(headDim)]
    )
    let key = QuantizedTensor(
      device: device,
      data: keyBuffer,
      parameters: asymmetricParams,
      elementCount: tensorSize,
      shape: [Int(seqLength), Int(headDim)]
    )
    let value = QuantizedTensor(
      device: device,
      data: valueBuffer,
      parameters: asymmetricParams,
      elementCount: tensorSize,
      shape: [Int(seqLength), Int(headDim)]
    )

    // Create gradient buffers (FP32)
    let gradOutputBuffer = device.makeBuffer(
      length: tensorSize * 4,
      options: .storageModeShared
    )!
    let logsumexpBuffer = device.makeBuffer(
      length: Int(seqLength) * 4,
      options: .storageModeShared
    )!
    let gradQueryBuffer = device.makeBuffer(
      length: tensorSize * 4,
      options: .storageModeShared
    )!
    let dValuesBuffer = device.makeBuffer(
      length: Int(seqLength) * 4,
      options: .storageModeShared
    )!

    // Create descriptor
    var descriptor = AttentionDescriptor()
    descriptor.lowPrecisionInputs = true
    descriptor.matrixDimensions = (row: seqLength, column: seqLength, head: headDim)
    descriptor.transposeState = (Q: false, K: false, V: false, O: false)
    descriptor.sparsityPattern = .none

    var config = QuantizedAttention.Configuration()
    config.queryPrecision = .INT8
    config.keyPrecision = .INT8
    config.valuePrecision = .INT8
    config.queryStrategy = .symmetric
    config.keyStrategy = .asymmetric
    config.valueStrategy = .asymmetric
    config.strategyVersion = 1

    let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
      baseDescriptor: descriptor,
      quantizationConfig: config
    )

    // Execute backward query
    let commandBuffer = quantizedAttention.backwardQuery(
      query: query,
      key: key,
      value: value,
      gradOutput: gradOutputBuffer,
      logsumexp: logsumexpBuffer,
      gradQuery: gradQueryBuffer,
      dValues: dValuesBuffer,
      descriptor: quantDescriptor
    )

    XCTAssertNotNil(commandBuffer, "Command buffer should be created for backward query")

    // Verify the encoding worked by checking command buffer completion
    commandBuffer?.commit()
    commandBuffer?.waitUntilCompleted()

    if let error = commandBuffer?.error {
      XCTFail("Command buffer execution error: \(error)")
    }
  }

  func testBackwardKeyValueStrategyEncoding() throws {
    // Test dimensions
    let seqLength: UInt32 = 32
    let headDim: UInt16 = 16
    let tensorSize = Int(seqLength * seqLength) * Int(headDim)

    // Create test tensors with mixed strategies
    let symmetricParams = QuantizationParameters(
      scale: 0.125,
      zeroPoint: 0,
      precision: .INT8,
      mode: .tensorWise,
      strategy: .symmetric,
      strategyVersion: 1
    )

    let asymmetricParams = QuantizationParameters(
      scale: 0.25,
      zeroPoint: 64,
      precision: .INT8,
      mode: .tensorWise,
      strategy: .asymmetric,
      strategyVersion: 1
    )

    // Create quantized tensors with different strategies
    let queryBuffer = device.makeBuffer(length: tensorSize, options: .storageModeShared)!
    let keyBuffer = device.makeBuffer(length: tensorSize, options: .storageModeShared)!
    let valueBuffer = device.makeBuffer(length: tensorSize, options: .storageModeShared)!

    let query = QuantizedTensor(
      device: device,
      data: queryBuffer,
      parameters: symmetricParams,
      elementCount: tensorSize,
      shape: [Int(seqLength), Int(headDim)]
    )
    let key = QuantizedTensor(
      device: device,
      data: keyBuffer,
      parameters: asymmetricParams,
      elementCount: tensorSize,
      shape: [Int(seqLength), Int(headDim)]
    )
    let value = QuantizedTensor(
      device: device,
      data: valueBuffer,
      parameters: symmetricParams, // Different strategy than key
      elementCount: tensorSize,
      shape: [Int(seqLength), Int(headDim)]
    )

    // Create gradient buffers
    let gradOutputBuffer = device.makeBuffer(
      length: tensorSize * 4,
      options: .storageModeShared
    )!
    let logsumexpBuffer = device.makeBuffer(
      length: Int(seqLength) * 4,
      options: .storageModeShared
    )!
    let dValuesBuffer = device.makeBuffer(
      length: Int(seqLength) * 4,
      options: .storageModeShared
    )!
    let gradKeyBuffer = device.makeBuffer(length: tensorSize * 4, options: .storageModeShared)!
    let gradValueBuffer = device.makeBuffer(
      length: tensorSize * 4,
      options: .storageModeShared
    )!

    // Create descriptor
    var descriptor = AttentionDescriptor()
    descriptor.lowPrecisionInputs = true
    descriptor.matrixDimensions = (row: seqLength, column: seqLength, head: headDim)
    descriptor.transposeState = (Q: false, K: false, V: false, O: false)
    descriptor.sparsityPattern = .none

    var config2 = QuantizedAttention.Configuration()
    config2.queryPrecision = .INT8
    config2.keyPrecision = .INT8
    config2.valuePrecision = .INT8
    config2.queryStrategy = .symmetric
    config2.keyStrategy = .asymmetric
    config2.valueStrategy = .symmetric
    config2.strategyVersion = 1

    let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
      baseDescriptor: descriptor,
      quantizationConfig: config2
    )

    // Execute backward key-value
    let commandBuffer = quantizedAttention.backwardKeyValue(
      query: query,
      key: key,
      value: value,
      gradOutput: gradOutputBuffer,
      logsumexp: logsumexpBuffer,
      dValues: dValuesBuffer,
      gradKey: gradKeyBuffer,
      gradValue: gradValueBuffer,
      descriptor: quantDescriptor
    )

    XCTAssertNotNil(commandBuffer, "Command buffer should be created for backward key-value")

    // Verify execution
    commandBuffer?.commit()
    commandBuffer?.waitUntilCompleted()

    if let error = commandBuffer?.error {
      XCTFail("Command buffer execution error: \(error)")
    }

    // Verify parameters match what was set
    XCTAssertEqual(query.parameters.strategy, QuantizationStrategy.symmetric)
    XCTAssertEqual(key.parameters.strategy, QuantizationStrategy.asymmetric)
    XCTAssertEqual(value.parameters.strategy, QuantizationStrategy.symmetric)
    XCTAssertEqual(query.parameters.strategyVersion, 1)
    XCTAssertEqual(key.parameters.strategyVersion, 1)
    XCTAssertEqual(value.parameters.strategyVersion, 1)
  }

  func testBackwardPassWithFutureStrategy() throws {
    // Test with a future strategy version to ensure extensibility
    let futureParams = QuantizationParameters(
      scale: 0.0625,
      zeroPoint: 0,
      precision: .INT8,
      mode: .tensorWise,
      strategy: .symmetric,
      strategyVersion: 255 // Max version
    )

    let seqLength: UInt32 = 16
    let headDim: UInt16 = 8
    let tensorSize = Int(seqLength * seqLength) * Int(headDim)

    let tensorBuffer = device.makeBuffer(length: tensorSize, options: .storageModeShared)!
    let tensor = QuantizedTensor(
      device: device,
      data: tensorBuffer,
      parameters: futureParams,
      elementCount: tensorSize,
      shape: [Int(seqLength), Int(headDim)]
    )

    // Create required buffers
    let gradBuffer = device.makeBuffer(length: tensorSize * 4, options: .storageModeShared)!
    let logsumexpBuffer = device.makeBuffer(
      length: Int(seqLength) * 4,
      options: .storageModeShared
    )!
    let gradQueryBuffer = device.makeBuffer(
      length: tensorSize * 4,
      options: .storageModeShared
    )!
    let dValuesBuffer = device.makeBuffer(
      length: Int(seqLength) * 4,
      options: .storageModeShared
    )!

    var descriptor = AttentionDescriptor()
    descriptor.lowPrecisionInputs = true
    descriptor.matrixDimensions = (row: seqLength, column: seqLength, head: headDim)
    descriptor.transposeState = (Q: false, K: false, V: false, O: false)
    descriptor.sparsityPattern = .none

    var config3 = QuantizedAttention.Configuration()
    config3.queryPrecision = .INT8
    config3.keyPrecision = .INT8
    config3.valuePrecision = .INT8
    config3.queryStrategy = .symmetric
    config3.keyStrategy = .symmetric
    config3.valueStrategy = .symmetric
    config3.strategyVersion = 255

    let quantDescriptor = QuantizedAttention.QuantizedAttentionDescriptor(
      baseDescriptor: descriptor,
      quantizationConfig: config3
    )

    let commandBuffer = quantizedAttention.backwardQuery(
      query: tensor,
      key: tensor,
      value: tensor,
      gradOutput: gradBuffer,
      logsumexp: logsumexpBuffer,
      gradQuery: gradQueryBuffer,
      dValues: dValuesBuffer,
      descriptor: quantDescriptor
    )

    XCTAssertNotNil(commandBuffer, "Should handle future strategy versions")
    commandBuffer?.commit()
    commandBuffer?.waitUntilCompleted()

    if let error = commandBuffer?.error {
      XCTFail("Command buffer error with future strategy: \(error)")
    }
  }
}

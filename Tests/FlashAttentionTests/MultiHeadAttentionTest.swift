//
//  MultiHeadAttentionTest.swift
//  FlashAttention
//
//  Created by bghira on 9/15/24.
//

import FlashAttention
import XCTest

final class MultiHeadAttentionTest: XCTestCase {

  private var device: MTLDevice!
  private var multiHeadAttention: MultiHeadAttention!

  override func setUp() {
    device = MTLContext.global.device
    multiHeadAttention = MultiHeadAttention(device: device)
  }

  // MARK: - Correctness Tests

  func testStandardMultiHeadAttention() throws {
    // Test standard MHA: Q, K, V all have [B, H, S, D] shape
    let batchSize: UInt32 = 2
    let numHeads: UInt32 = 8
    let sequenceLength: UInt32 = 64
    let headDimension: UInt16 = 32

    let queryShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    validateMultiHeadAttention(
      queryShape: queryShape,
      keyShape: queryShape,
      valueShape: queryShape,
      broadcastMode: .standard,
      testName: "Standard MHA"
    )
  }

  func testGroupedQueryAttention() throws {
    // Test GQA: Q has [B, H, S, D], K/V have [B, H_kv, S, D] where H_kv < H
    let batchSize: UInt32 = 2
    let numHeads: UInt32 = 8
    let numKVHeads: UInt32 = 2
    let sequenceLength: UInt32 = 64
    let headDimension: UInt16 = 32

    let queryShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    let kvShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numKVHeads,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    validateMultiHeadAttention(
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape,
      broadcastMode: .groupedQuery(numKVHeads: numKVHeads),
      testName: "Grouped Query Attention"
    )
  }

  func testMultiQueryAttention() throws {
    // Test MQA: Q has [B, H, S, D], K/V have [B, 1, S, D]
    let batchSize: UInt32 = 2
    let numHeads: UInt32 = 8
    let sequenceLength: UInt32 = 64
    let headDimension: UInt16 = 32

    let queryShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    let kvShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: 1,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    validateMultiHeadAttention(
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape,
      broadcastMode: .multiQuery,
      testName: "Multi-Query Attention"
    )
  }

  func testCrossAttention() throws {
    // Test cross-attention: Q from [B, H, S_q, D], K/V from [B, H, S_kv, D]
    let batchSize: UInt32 = 2
    let numHeads: UInt32 = 8
    let querySeqLen: UInt32 = 32
    let kvSeqLen: UInt32 = 128
    let headDimension: UInt16 = 32

    let queryShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: querySeqLen,
      headDimension: headDimension
    )

    let kvShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: kvSeqLen,
      headDimension: headDimension
    )

    validateMultiHeadAttention(
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape,
      broadcastMode: .crossAttention(kvSequenceLength: kvSeqLen),
      testName: "Cross-Attention"
    )
  }

  // MARK: - Performance Tests

  func testPerformanceStandardMHA() throws {
    measurePerformanceForConfiguration(
      batchSize: 4,
      numHeads: 8,
      sequenceLength: 512,
      headDimension: 64,
      broadcastMode: .standard,
      testName: "Standard MHA Performance"
    )
  }

  func testPerformanceGroupedQuery() throws {
    measurePerformanceForConfiguration(
      batchSize: 4,
      numHeads: 32,
      numKVHeads: 8,
      sequenceLength: 512,
      headDimension: 64,
      testName: "GQA Performance"
    )
  }

  func testPerformanceMultiQuery() throws {
    measurePerformanceForConfiguration(
      batchSize: 4,
      numHeads: 32,
      numKVHeads: 1,
      sequenceLength: 512,
      headDimension: 64,
      testName: "MQA Performance"
    )
  }

  // MARK: - Broadcast Semantics Tests

  func testBroadcastValidation() throws {
    let batchSize: UInt32 = 2
    let numHeads: UInt32 = 8
    let sequenceLength: UInt32 = 64
    let headDimension: UInt16 = 32

    let standardShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    // Test valid broadcast modes
    XCTAssertTrue(
      MultiHeadBroadcastMode.standard.isCompatible(
        qShape: standardShape,
        kShape: standardShape,
        vShape: standardShape
      )
    )

    let gqaKVShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: 2,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    XCTAssertTrue(
      MultiHeadBroadcastMode.groupedQuery(numKVHeads: 2).isCompatible(
        qShape: standardShape,
        kShape: gqaKVShape,
        vShape: gqaKVShape
      )
    )

    let mqaKVShape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: 1,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    XCTAssertTrue(
      MultiHeadBroadcastMode.multiQuery.isCompatible(
        qShape: standardShape,
        kShape: mqaKVShape,
        vShape: mqaKVShape
      )
    )

    // Test invalid broadcast modes
    XCTAssertFalse(
      MultiHeadBroadcastMode.standard.isCompatible(
        qShape: standardShape,
        kShape: gqaKVShape,
        vShape: gqaKVShape
      )
    )
  }

  // MARK: - Dispatch Strategy Tests

  func testDispatchStrategies() throws {
    let batchSize: UInt32 = 2
    let numHeads: UInt32 = 4
    let sequenceLength: UInt32 = 32
    let headDimension: UInt16 = 16

    let shape = MultiHeadShape(
      batchSize: batchSize,
      numHeads: numHeads,
      sequenceLength: sequenceLength,
      headDimension: headDimension
    )

    let strategies: [MultiHeadDispatchStrategy] = [
      .perBatchHead,
      .perBatch,
      .batched
    ]

    for strategy in strategies {
      validateMultiHeadAttention(
        queryShape: shape,
        keyShape: shape,
        valueShape: shape,
        broadcastMode: .standard,
        dispatchStrategy: strategy,
        testName: "Dispatch Strategy: \(strategy)"
      )
    }
  }

  // MARK: - Microbenchmark Tests

  func testMicrobenchmarkCorrectness() throws {
    // Small problem sizes for detailed correctness validation
    let configs = [
      (batchSize: 1, numHeads: 1, seqLen: 8, headDim: 4),
      (batchSize: 1, numHeads: 2, seqLen: 16, headDim: 8),
      (batchSize: 2, numHeads: 4, seqLen: 32, headDim: 16),
    ]

    for config in configs {
      let shape = MultiHeadShape(
        batchSize: UInt32(config.batchSize),
        numHeads: UInt32(config.numHeads),
        sequenceLength: UInt32(config.seqLen),
        headDimension: UInt16(config.headDim)
      )

      validateMultiHeadAttention(
        queryShape: shape,
        keyShape: shape,
        valueShape: shape,
        broadcastMode: .standard,
        testName: "Microbenchmark \(config.batchSize)x\(config.numHeads)x\(config.seqLen)x\(config.headDim)"
      )
    }
  }

  func testMicrobenchmarkFlexibility() throws {
    // Test various combinations of batch/head dimensions
    let flexibilityConfigs = [
      // Different batch sizes
      (batchSize: 1, numHeads: 4, numKVHeads: 4),
      (batchSize: 3, numHeads: 4, numKVHeads: 4),
      (batchSize: 7, numHeads: 4, numKVHeads: 4),

      // Different head ratios for GQA
      (batchSize: 2, numHeads: 8, numKVHeads: 2),
      (batchSize: 2, numHeads: 12, numKVHeads: 3),
      (batchSize: 2, numHeads: 16, numKVHeads: 4),

      // MQA scenarios
      (batchSize: 2, numHeads: 8, numKVHeads: 1),
      (batchSize: 4, numHeads: 16, numKVHeads: 1),
    ]

    for config in flexibilityConfigs {
      let queryShape = MultiHeadShape(
        batchSize: UInt32(config.batchSize),
        numHeads: UInt32(config.numHeads),
        sequenceLength: 32,
        headDimension: 16
      )

      let kvShape = MultiHeadShape(
        batchSize: UInt32(config.batchSize),
        numHeads: UInt32(config.numKVHeads),
        sequenceLength: 32,
        headDimension: 16
      )

      let broadcastMode: MultiHeadBroadcastMode
      if config.numKVHeads == config.numHeads {
        broadcastMode = .standard
      } else if config.numKVHeads == 1 {
        broadcastMode = .multiQuery
      } else {
        broadcastMode = .groupedQuery(numKVHeads: UInt32(config.numKVHeads))
      }

      validateMultiHeadAttention(
        queryShape: queryShape,
        keyShape: kvShape,
        valueShape: kvShape,
        broadcastMode: broadcastMode,
        testName: "Flexibility B\(config.batchSize)_H\(config.numHeads)_KV\(config.numKVHeads)"
      )
    }
  }

  func testMicrobenchmarkPerformance() throws {
    // Performance comparison between broadcast modes
    let batchSize: UInt32 = 4
    let sequenceLength: UInt32 = 256
    let headDimension: UInt16 = 64

    var results: [String: Double] = [:]

    // Standard MHA
    let standardTime = measureExecutionTime {
      measurePerformanceForConfiguration(
        batchSize: Int(batchSize),
        numHeads: 8,
        sequenceLength: Int(sequenceLength),
        headDimension: Int(headDimension),
        broadcastMode: .standard,
        testName: "Perf Standard",
        silent: true
      )
    }
    results["Standard_MHA"] = standardTime

    // GQA
    let gqaTime = measureExecutionTime {
      measurePerformanceForConfiguration(
        batchSize: Int(batchSize),
        numHeads: 8,
        numKVHeads: 2,
        sequenceLength: Int(sequenceLength),
        headDimension: Int(headDimension),
        testName: "Perf GQA",
        silent: true
      )
    }
    results["GQA"] = gqaTime

    // MQA
    let mqaTime = measureExecutionTime {
      measurePerformanceForConfiguration(
        batchSize: Int(batchSize),
        numHeads: 8,
        numKVHeads: 1,
        sequenceLength: Int(sequenceLength),
        headDimension: Int(headDimension),
        testName: "Perf MQA",
        silent: true
      )
    }
    results["MQA"] = mqaTime

    // Print performance comparison
    print("\nPerformance Comparison (ms):")
    for (mode, time) in results.sorted(by: { $0.value < $1.value }) {
      print("  \(mode): \(String(format: "%.3f", time * 1000))")
    }

    // Verify expected performance ordering: MQA <= GQA <= Standard
    XCTAssertLessThanOrEqual(results["MQA"]!, results["GQA"]! * 1.2, "MQA should be faster than or similar to GQA")
    XCTAssertLessThanOrEqual(results["GQA"]!, results["Standard_MHA"]! * 1.1, "GQA should be faster than or similar to Standard MHA")
  }

  // MARK: - Helper Methods

  private func validateMultiHeadAttention(
    queryShape: MultiHeadShape,
    keyShape: MultiHeadShape,
    valueShape: MultiHeadShape,
    broadcastMode: MultiHeadBroadcastMode,
    dispatchStrategy: MultiHeadDispatchStrategy = .auto,
    testName: String
  ) {
    print("Testing: \(testName)")

    // Create base descriptor
    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.lowPrecisionInputs = false
    baseDescriptor.lowPrecisionIntermediates = false
    baseDescriptor.matrixDimensions = (
      row: queryShape.sequenceLength,
      column: keyShape.sequenceLength,
      head: queryShape.headDimension
    )
    baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

    // Create multi-head descriptor
    let descriptor = MultiHeadAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      queryShape: queryShape,
      keyShape: keyShape,
      valueShape: valueShape,
      broadcastMode: broadcastMode,
      dispatchStrategy: dispatchStrategy
    )

    // Create test data
    let (queryBuffer, keyBuffer, valueBuffer, outputBuffer, logsumexpBuffer) = createTestBuffers(
      queryShape: queryShape,
      keyShape: keyShape,
      valueShape: valueShape
    )

    // Execute forward pass
    guard let commandBuffer = multiHeadAttention.forward(
      query: queryBuffer,
      key: keyBuffer,
      value: valueBuffer,
      output: outputBuffer,
      logsumexp: logsumexpBuffer,
      descriptor: descriptor
    ) else {
      XCTFail("Failed to create command buffer for \(testName)")
      return
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Validate results
    validateResults(
      outputBuffer: outputBuffer,
      logsumexpBuffer: logsumexpBuffer,
      descriptor: descriptor,
      testName: testName
    )

    print("✓ \(testName) passed")
  }

  private func measurePerformanceForConfiguration(
    batchSize: Int,
    numHeads: Int,
    numKVHeads: Int? = nil,
    sequenceLength: Int,
    headDimension: Int,
    broadcastMode: MultiHeadBroadcastMode? = nil,
    testName: String,
    silent: Bool = false
  ) {
    let actualNumKVHeads = numKVHeads ?? numHeads
    let actualBroadcastMode = broadcastMode ?? {
      if actualNumKVHeads == numHeads {
        return .standard
      } else if actualNumKVHeads == 1 {
        return .multiQuery
      } else {
        return .groupedQuery(numKVHeads: UInt32(actualNumKVHeads))
      }
    }()

    let queryShape = MultiHeadShape(
      batchSize: UInt32(batchSize),
      numHeads: UInt32(numHeads),
      sequenceLength: UInt32(sequenceLength),
      headDimension: UInt16(headDimension)
    )

    let kvShape = MultiHeadShape(
      batchSize: UInt32(batchSize),
      numHeads: UInt32(actualNumKVHeads),
      sequenceLength: UInt32(sequenceLength),
      headDimension: UInt16(headDimension)
    )

    var baseDescriptor = AttentionDescriptor()
    baseDescriptor.lowPrecisionInputs = false
    baseDescriptor.lowPrecisionIntermediates = false
    baseDescriptor.matrixDimensions = (
      row: queryShape.sequenceLength,
      column: kvShape.sequenceLength,
      head: queryShape.headDimension
    )
    baseDescriptor.transposeState = (Q: false, K: false, V: false, O: false)

    let descriptor = MultiHeadAttentionDescriptor(
      baseDescriptor: baseDescriptor,
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape,
      broadcastMode: actualBroadcastMode
    )

    let (queryBuffer, keyBuffer, valueBuffer, outputBuffer, logsumexpBuffer) = createTestBuffers(
      queryShape: queryShape,
      keyShape: kvShape,
      valueShape: kvShape
    )

    // Warmup
    for _ in 0..<10 {
      if let commandBuffer = multiHeadAttention.forward(
        query: queryBuffer,
        key: keyBuffer,
        value: valueBuffer,
        output: outputBuffer,
        logsumexp: logsumexpBuffer,
        descriptor: descriptor
      ) {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
      }
    }

    // Benchmark
    let iterations = 100
    let startTime = CFAbsoluteTimeGetCurrent()

    for _ in 0..<iterations {
      if let commandBuffer = multiHeadAttention.forward(
        query: queryBuffer,
        key: keyBuffer,
        value: valueBuffer,
        output: outputBuffer,
        logsumexp: logsumexpBuffer,
        descriptor: descriptor
      ) {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
      }
    }

    let endTime = CFAbsoluteTimeGetCurrent()
    let avgTime = (endTime - startTime) / Double(iterations)

    // Calculate throughput
    let totalOps = 2.0 * Double(batchSize) * Double(numHeads) * Double(sequenceLength) * Double(sequenceLength) * Double(headDimension)
    let gops = totalOps / (avgTime * 1e9)

    if !silent {
      print("\(testName) Performance:")
      print("  Average time: \(String(format: "%.3f", avgTime * 1000)) ms")
      print("  Throughput: \(String(format: "%.1f", gops)) GOPS")
    }
  }

  private func createTestBuffers(
    queryShape: MultiHeadShape,
    keyShape: MultiHeadShape,
    valueShape: MultiHeadShape
  ) -> (MTLBuffer, MTLBuffer, MTLBuffer, MTLBuffer, MTLBuffer) {
    // Create random test data
    let queryData = generateRandomData(count: Int(queryShape.totalElements))
    let keyData = generateRandomData(count: Int(keyShape.totalElements))
    let valueData = generateRandomData(count: Int(valueShape.totalElements))

    guard
      let queryBuffer = device.makeBuffer(bytes: queryData, length: queryData.count * MemoryLayout<Float>.size),
      let keyBuffer = device.makeBuffer(bytes: keyData, length: keyData.count * MemoryLayout<Float>.size),
      let valueBuffer = device.makeBuffer(bytes: valueData, length: valueData.count * MemoryLayout<Float>.size),
      let outputBuffer = device.makeBuffer(length: Int(queryShape.totalElements) * MemoryLayout<Float>.size),
      let logsumexpBuffer = device.makeBuffer(length: Int(queryShape.batchSize * queryShape.numHeads * queryShape.sequenceLength) * MemoryLayout<Float>.size)
    else {
      fatalError("Failed to create Metal buffers")
    }

    return (queryBuffer, keyBuffer, valueBuffer, outputBuffer, logsumexpBuffer)
  }

  private func generateRandomData(count: Int) -> [Float] {
    (0..<count).map { _ in Float.random(in: -1...1) }
  }

  private func validateResults(
    outputBuffer: MTLBuffer,
    logsumexpBuffer: MTLBuffer,
    descriptor: MultiHeadAttentionDescriptor,
    testName: String
  ) {
    // Basic validation: check for NaN/Inf values
    let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: Int(descriptor.queryShape.totalElements))
    let logsumexpPointer = logsumexpBuffer.contents().bindMemory(to: Float.self, capacity: Int(descriptor.queryShape.batchSize * descriptor.queryShape.numHeads * descriptor.queryShape.sequenceLength))

    for i in 0..<Int(descriptor.queryShape.totalElements) {
      let value = outputPointer[i]
      XCTAssertFalse(value.isNaN, "Output contains NaN at index \(i) in \(testName)")
      XCTAssertFalse(value.isInfinite, "Output contains Inf at index \(i) in \(testName)")
    }

    for i in 0..<Int(descriptor.queryShape.batchSize * descriptor.queryShape.numHeads * descriptor.queryShape.sequenceLength) {
      let value = logsumexpPointer[i]
      XCTAssertFalse(value.isNaN, "Logsumexp contains NaN at index \(i) in \(testName)")
      XCTAssertFalse(value.isInfinite, "Logsumexp contains Inf at index \(i) in \(testName)")
    }
  }

  private func measureExecutionTime(_ block: () -> Void) -> Double {
    let startTime = CFAbsoluteTimeGetCurrent()
    block()
    let endTime = CFAbsoluteTimeGetCurrent()
    return endTime - startTime
  }
}
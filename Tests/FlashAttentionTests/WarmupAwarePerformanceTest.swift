import FlashAttention
import XCTest

final class WarmupAwarePerformanceTest: XCTestCase {
  struct WarmupResult {
    let sequenceLength: Int
    let headDimension: Int
    let coldStartNormal: Double
    let coldStartCausal: Double
    let warmNormal: Double
    let warmCausal: Double
    let coldSpeedup: Double
    let warmSpeedup: Double
    let warmupBenefit: Double // How much warmup improves performance

    var description: String {
      """
      seq=\(sequenceLength), head=\(headDimension):
        Cold: Normal=\(String(format: "%.3f", coldStartNormal))ms, Causal=\(String(format: "%.3f",
                                                                                   coldStartCausal))ms, Speedup=\(
        String(
          format: "%.1f%%",
          (
            coldSpeedup -
              1.0
          ) * 100
        )
      )
        Warm: Normal=\(String(format: "%.3f", warmNormal))ms, Causal=\(String(format: "%.3f",
                                                                              warmCausal))ms, Speedup=\(
        String(
          format: "%.1f%%",
          (
            warmSpeedup - 1.0
          ) *
            100
        )
      )
        Warmup benefit: \(String(format: "%.1f%%", warmupBenefit * 100))
      """
    }
  }

  func testCalibrationPayloadRoundTrip() throws {
    let heuristic = MaskingStrategyHeuristic.shared
    heuristic.clearCache()

    // Calibrate a couple of small shapes. calibrate() returns a serializable
    // payload AND populates the in-memory cache as a side effect.
    let shapes = [
      (sequenceLength: 256, headDimension: 64),
      (sequenceLength: 512, headDimension: 128),
    ]
    let calibration = heuristic.calibrate(shapes: shapes, warmupIterations: 3, trialIterations: 5)

    XCTAssertEqual(
      calibration.entries.count,
      shapes.count,
      "calibrate should produce one entry per shape"
    )
    XCTAssertEqual(calibration.deviceName, MTLContext.global.device.name)
    print("\n📦 Calibration payload (\(calibration.deviceName)):")
    for entry in calibration.entries {
      print(
        "  seq~\(entry.sequenceBucket), head=\(entry.headDimension): " +
          "\(entry.strategy == .bitmask ? "BITMASK" : "ELEMENT-WISE") " +
          "(bitmask \(String(format: "%.3f", entry.bitmaskMs))ms vs " +
          "element-wise \(String(format: "%.3f", entry.elementWiseMs))ms)"
      )
    }

    // Persist to a temp file via the separate store, then load it back and
    // hydrate a fresh cache. Demonstrates the decoupled persistence: callers
    // who don't persist just skip this and use the in-memory cache.
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("masking-calibration-test.json")
    try? FileManager.default.removeItem(at: url)
    try MaskingCalibrationStore.save(calibration, to: url)
    let loaded = try MaskingCalibrationStore.load(from: url)

    heuristic.clearCache()

    // After apply(), recommend() must return the measured strategy for each
    // calibrated shape regardless of the default rule.
    heuristic.apply(loaded)
    for (shape, entry) in zip(shapes, loaded.entries) {
      let recommended = heuristic.recommend(
        sequenceLength: shape.sequenceLength,
        headDimension: shape.headDimension
      )
      XCTAssertEqual(recommended, entry.strategy, "applied calibration should drive recommend()")
    }

    try? FileManager.default.removeItem(at: url)
    print("✅ calibrate → save → load → apply round-trip verified")
  }

  func testWarmUpAmortizesAcrossRuns() throws {
    let heuristic = MaskingStrategyHeuristic.shared
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("masking-warmup-test.json")
    try? FileManager.default.removeItem(at: url)

    let shapes = [(sequenceLength: 256, headDimension: 64)]

    // First warmUp: no persisted file → benchmark, save, apply.
    heuristic.clearCache()
    let first = heuristic.warmUp(
      shapes: shapes,
      persistTo: url,
      warmupIterations: 3,
      trialIterations: 5
    )
    XCTAssertFalse(first.entries.isEmpty, "first warmUp should calibrate")
    XCTAssertTrue(
      FileManager.default.fileExists(atPath: url.path),
      "warmUp should persist when given a URL"
    )
    let firstStrategy = first.entries[0].strategy
    XCTAssertEqual(
      heuristic.recommend(sequenceLength: 256, headDimension: 64), firstStrategy,
      "warmUp should populate the cache"
    )

    // Second warmUp: file exists + device matches → load (no benchmark).
    // Proven by the cache being repopulated after a clear WITHOUT calibrating:
    // recommend() still returns the persisted strategy.
    heuristic.clearCache()
    let second = heuristic.warmUp(
      shapes: shapes,
      persistTo: url,
      warmupIterations: 3,
      trialIterations: 5
    )
    XCTAssertEqual(second.deviceName, first.deviceName)
    XCTAssertEqual(second.entries[0].strategy, firstStrategy, "loaded calibration should match")
    XCTAssertEqual(
      heuristic.recommend(sequenceLength: 256, headDimension: 64), firstStrategy,
      "loaded calibration should drive recommend()"
    )

    try? FileManager.default.removeItem(at: url)
    print("✅ warmUp amortizes: first call calibrated+persisted, second call loaded")
  }

  func testWarmupAwareAutoOptimization() throws {
    print("\n🔥 Warmup-Aware Auto-Optimization Validation")
    print("=" + String(repeating: "=", count: 80))

    // Test realistic workload sizes that might need warmup consideration
    let testCases = [
      (seq: 512, head: 64), // Small-medium
      (seq: 1024, head: 64), // Medium
      (seq: 1024, head: 128), // Medium-large head
      (seq: 2048, head: 64), // Large
      (seq: 1536, head: 128), // Large + large head
    ]

    var warmupResults: [WarmupResult] = []

    for (seq, head) in testCases {
      print("\n🧪 Testing seq=\(seq), head=\(head) with proper warmup...")

      // Test cold start performance
      let (coldNormal, coldCausal) = measureColdStartPerformance(
        sequenceDimension: seq, headDimension: head
      )

      // Test warm performance (after proper warmup)
      let (warmNormal, warmCausal) = measureWarmPerformance(
        sequenceDimension: seq, headDimension: head
      )

      let coldSpeedup = coldNormal / coldCausal
      let warmSpeedup = warmNormal / warmCausal
      let warmupBenefit = (coldCausal - warmCausal) / coldCausal

      let result = WarmupResult(
        sequenceLength: seq,
        headDimension: head,
        coldStartNormal: coldNormal,
        coldStartCausal: coldCausal,
        warmNormal: warmNormal,
        warmCausal: warmCausal,
        coldSpeedup: coldSpeedup,
        warmSpeedup: warmSpeedup,
        warmupBenefit: warmupBenefit
      )

      warmupResults.append(result)
      print(result.description)
    }

    analyzeWarmupResults(warmupResults)
    testAutoOptimizationDecisions(warmupResults)
  }

  private func measureColdStartPerformance(
    sequenceDimension: Int,
    headDimension: Int
  )
    -> (normal: Double, causal: Double)
  {
    // Simulate cold start by creating fresh pipeline states

    let normalTime = measureSingleColdRun(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      usesCausal: false
    )

    let causalTime = measureSingleColdRun(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      usesCausal: true
    )

    return (normal: normalTime, causal: causalTime)
  }

  private func measureWarmPerformance(
    sequenceDimension: Int,
    headDimension: Int
  )
    -> (normal: Double, causal: Double)
  {
    // Proper warmup followed by accurate measurement

    let normalTime = measureWithProperWarmup(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      usesCausal: false
    )

    let causalTime = measureWithProperWarmup(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      usesCausal: true
    )

    return (normal: normalTime, causal: causalTime)
  }

  private func measureSingleColdRun(
    sequenceDimension: Int,
    headDimension: Int,
    usesCausal: Bool
  )
    -> Double
  {
    // Fresh pipeline state for each measurement
    let (pipeline, buffers) = createFreshPipelineAndBuffers(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      usesCausal: usesCausal
    )

    // Single cold execution
    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!

    let startTime = CACurrentMediaTime()

    encoder.setComputePipelineState(pipeline.pipeline)
    setBuffers(encoder: encoder, buffers: buffers)
    dispatchKernel(encoder: encoder, pipeline: pipeline, sequenceDimension: sequenceDimension)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let endTime = CACurrentMediaTime()
    return (endTime - startTime) * 1000.0
  }

  private func measureWithProperWarmup(
    sequenceDimension: Int,
    headDimension: Int,
    usesCausal: Bool
  )
    -> Double
  {
    let (pipeline, buffers) = createFreshPipelineAndBuffers(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      usesCausal: usesCausal
    )

    // Extensive warmup (like real workloads)
    for _ in 0..<20 {
      let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
      let encoder = commandBuffer.makeComputeCommandEncoder()!

      encoder.setComputePipelineState(pipeline.pipeline)
      setBuffers(encoder: encoder, buffers: buffers)
      dispatchKernel(encoder: encoder, pipeline: pipeline, sequenceDimension: sequenceDimension)
      encoder.endEncoding()

      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }

    // Now measure performance with warmed-up state
    let iterations = 50
    let startTime = CACurrentMediaTime()

    for _ in 0..<iterations {
      let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
      let encoder = commandBuffer.makeComputeCommandEncoder()!

      encoder.setComputePipelineState(pipeline.pipeline)
      setBuffers(encoder: encoder, buffers: buffers)
      dispatchKernel(encoder: encoder, pipeline: pipeline, sequenceDimension: sequenceDimension)
      encoder.endEncoding()

      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }

    let endTime = CACurrentMediaTime()
    return (endTime - startTime) * 1000.0 / Double(iterations)
  }

  private func createFreshPipelineAndBuffers(
    sequenceDimension: Int,
    headDimension: Int,
    usesCausal: Bool
  )
    -> (pipeline: (
      pipeline: MTLComputePipelineState,
      blockDimensions: (parallelization: UInt16, traversal: UInt16, head: UInt16)
    ), buffers: (Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, O: MTLBuffer, L: MTLBuffer))
  {
    var attentionDesc = AttentionDescriptor()
    attentionDesc.lowPrecisionInputs = false
    attentionDesc.lowPrecisionIntermediates = false
    attentionDesc.matrixDimensions = (
      row: UInt32(sequenceDimension),
      column: UInt32(sequenceDimension),
      head: UInt16(headDimension)
    )
    attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
    attentionDesc.sparsityPattern = usesCausal ? .causal : .none

    let forwardDesc = attentionDesc.kernelDescriptor(type: .forward)
    let forwardKernel = AttentionKernel(descriptor: forwardDesc)
    let forwardSource = forwardKernel.createSource()

    let device = MTLContext.global.device
    let library = try! device.makeLibrary(source: forwardSource, options: nil)

    let functionConstants = MTLFunctionConstantValues()
    let function = try! library.makeFunction(
      name: "attention", constantValues: functionConstants
    )

    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
    let pipeline = try! device.makeComputePipelineState(
      descriptor: pipelineDesc, options: [], reflection: nil
    )

    // Create buffers
    let elementCount = sequenceDimension * headDimension
    let Q = device.makeBuffer(length: elementCount * 4, options: [])!
    let K = device.makeBuffer(length: elementCount * 4, options: [])!
    let V = device.makeBuffer(length: elementCount * 4, options: [])!
    let O = device.makeBuffer(length: elementCount * 4, options: [])!
    let L = device.makeBuffer(length: sequenceDimension * 4, options: [])!

    // Initialize with meaningful data
    initializeBuffer(Q, count: elementCount)
    initializeBuffer(K, count: elementCount)
    initializeBuffer(V, count: elementCount)

    let blockDimensions = forwardDesc.blockDimensions!
    return (
      pipeline: (pipeline: pipeline, blockDimensions: blockDimensions),
      buffers: (Q: Q, K: K, V: V, O: O, L: L)
    )
  }

  private func setBuffers(
    encoder: MTLComputeCommandEncoder,
    buffers: (Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer, O: MTLBuffer, L: MTLBuffer)
  ) {
    encoder.setBuffer(buffers.Q, offset: 0, index: 0)
    encoder.setBuffer(buffers.K, offset: 0, index: 1)
    encoder.setBuffer(buffers.V, offset: 0, index: 2)
    encoder.setBuffer(buffers.O, offset: 0, index: 3)
    encoder.setBuffer(buffers.L, offset: 0, index: 4)
  }

  private func dispatchKernel(
    encoder: MTLComputeCommandEncoder,
    pipeline: (
      pipeline: MTLComputePipelineState,
      blockDimensions: (parallelization: UInt16, traversal: UInt16, head: UInt16)
    ),
    sequenceDimension: Int
  ) {
    let threadsPerThreadgroup = MTLSize(
      width: Int(pipeline.blockDimensions.parallelization), height: 1, depth: 1
    )
    let threadgroupsPerGrid = MTLSize(
      width: (sequenceDimension + Int(pipeline.blockDimensions.parallelization) - 1) /
        Int(pipeline.blockDimensions.parallelization),
      height: 1, depth: 1
    )

    encoder.dispatchThreadgroups(
      threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup
    )
  }

  private func initializeBuffer(_ buffer: MTLBuffer, count: Int) {
    let data = buffer.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count {
      data[i] = Float.random(in: -1...1)
    }
  }

  private func analyzeWarmupResults(_ results: [WarmupResult]) {
    print("\n📊 Warmup Analysis")
    print("-" + String(repeating: "-", count: 60))

    let avgWarmupBenefit = results.map(\.warmupBenefit).reduce(0, +) / Double(results.count)
    print("Average warmup benefit: \(String(format: "%.1f%%", avgWarmupBenefit * 100))")

    let coldSpeedups = results.map(\.coldSpeedup)
    let warmSpeedups = results.map(\.warmSpeedup)

    let avgColdSpeedup = coldSpeedups.reduce(0, +) / Double(coldSpeedups.count)
    let avgWarmSpeedup = warmSpeedups.reduce(0, +) / Double(warmSpeedups.count)

    print("Average cold speedup: \(String(format: "%+.1f%%", (avgColdSpeedup - 1.0) * 100))")
    print("Average warm speedup: \(String(format: "%+.1f%%", (avgWarmSpeedup - 1.0) * 100))")

    // Check if warmup changes the optimization decisions
    let coldBetter = results.filter { $0.coldSpeedup > 1.0 }.count
    let warmBetter = results.filter { $0.warmSpeedup > 1.0 }.count

    print("\nOptimization decision consistency:")
    print("Cold start favors bitmask: \(coldBetter)/\(results.count) cases")
    print("Warm favors bitmask: \(warmBetter)/\(results.count) cases")

    if coldBetter != warmBetter {
      print("⚠️  Warning: Warmup changes optimization decisions!")
    } else {
      print("✅ Optimization decisions are consistent with warmup")
    }
  }

  private func testAutoOptimizationDecisions(_ results: [WarmupResult]) {
    print("\n🤖 Shared heuristic vs. measured (warm)")
    print("-" + String(repeating: "-", count: 60))

    // Clear any prior calibration so we measure the DEFAULT rule's accuracy.
    MaskingStrategyHeuristic.shared.clearCache()

    var ruleCorrect = 0
    for result in results {
      let seq = result.sequenceLength
      let head = result.headDimension
      let decision = MaskingStrategyHeuristic.shared.recommend(
        sequenceLength: seq, headDimension: head
      )
      let actuallyBitmask = result.warmSpeedup > 1.0
      let actual: MaskingStrategy = actuallyBitmask ? .bitmask : .elementWise
      let correct = (decision == actual)
      if correct { ruleCorrect += 1 }

      print(
        "seq=\(seq), head=\(head): rule=\(decision == .bitmask ? "BITMASK     " : "ELEMENT-WISE"), " +
          "measured=\(actual == .bitmask ? "BITMASK     " : "ELEMENT-WISE") " +
          "(warm \(String(format: "%+.1f%%", result.warmSpeedup * 100 - 100))) \(correct ? "✅" : "❌")"
      )

      // Feed the warm measurement back into the shared cache so the kernel
      // and downstream callers see the calibrated truth for this shape.
      MaskingStrategyHeuristic.shared.recordMeasurement(
        sequenceLength: seq, headDimension: head, strategy: actual
      )
    }

    print(
      "\nDefault-rule accuracy: \(ruleCorrect)/\(results.count) " +
        "(\(String(format: "%.0f%%", Double(ruleCorrect) / Double(results.count) * 100)))"
    )

    // Confirm the cache now reflects the measured truth.
    var cacheMatches = 0
    for result in results {
      let cached = MaskingStrategyHeuristic.shared.recommend(
        sequenceLength: result.sequenceLength, headDimension: result.headDimension
      )
      let actual: MaskingStrategy = result.warmSpeedup > 1.0 ? .bitmask : .elementWise
      if cached == actual { cacheMatches += 1 }
    }
    print("Post-calibration cache accuracy: \(cacheMatches)/\(results.count)")
  }
}

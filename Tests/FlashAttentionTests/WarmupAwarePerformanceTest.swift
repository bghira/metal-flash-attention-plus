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
    print("\n🤖 Testing Auto-Optimization Decisions")
    print("-" + String(repeating: "-", count: 60))

    for result in results {
      // Test our auto-optimization heuristic
      let totalElements = result.sequenceLength * result.sequenceLength * result.headDimension
      let shouldUseBitmask = shouldUseBitmaskOptimization(
        sequenceLength: result.sequenceLength,
        headDimension: result.headDimension,
        totalElements: totalElements
      )

      let actuallyBetter = result.warmSpeedup > 1.0
      let decision = shouldUseBitmask ? "BITMASK" : "ELEMENT-WISE"
      let correct = shouldUseBitmask == actuallyBetter

      print(
        "seq=\(result.sequenceLength), head=\(result.headDimension): " +
          "Decision=\(decision), Actually better=\(actuallyBetter ? "BITMASK" : "ELEMENT-WISE") " +
          "\(correct ? "✅" : "❌")"
      )
    }
  }

  // Copy of the auto-optimization logic from the main implementation
  private func shouldUseBitmaskOptimization(
    sequenceLength: Int,
    headDimension: Int,
    totalElements: Int
  )
    -> Bool
  {
    if totalElements < 50_331_648 {
      return true
    }

    if headDimension == 64 || headDimension == 128 {
      return true
    }

    if sequenceLength <= 256 {
      return true
    }

    return false
  }
}

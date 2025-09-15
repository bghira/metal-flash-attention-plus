import FlashAttention
import XCTest

final class CrossoverBenchmarkTest: XCTestCase {
  struct BenchmarkResult {
    let sequenceLength: Int
    let headDimension: Int
    let normalTime: Double
    let causalTime: Double
    let speedup: Double
    let isBitmaskFaster: Bool

    var description: String {
      let speedupStr = isBitmaskFaster ?
        String(format: "+%.1f%%", (speedup - 1.0) * 100) :
        String(format: "%.1f%%", (speedup - 1.0) * 100)
      return "seq=\(sequenceLength), head=\(headDimension): \(speedupStr) \(isBitmaskFaster ? "🚀" : "❌")"
    }
  }

  func testComprehensiveCrossoverAnalysis() throws {
    print("\n🔬 Comprehensive Crossover Point Analysis")
    print("=" + String(repeating: "=", count: 80))

    // Comprehensive test matrix
    let sequenceLengths = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
    let headDimensions = [32, 64, 96, 128, 160, 192, 256]

    var results: [BenchmarkResult] = []
    var fastRegions: [(seq: Int, head: Int)] = []
    var slowRegions: [(seq: Int, head: Int)] = []

    print("\n📊 Running \(sequenceLengths.count * headDimensions.count) benchmark combinations...")

    for seq in sequenceLengths {
      for head in headDimensions {
        print("Testing seq=\(seq), head=\(head)...", terminator: " ")

        let normalTime = measureOptimizedAttentionTime(
          sequenceDimension: seq, headDimension: head, usesCausal: false
        )
        let causalTime = measureOptimizedAttentionTime(
          sequenceDimension: seq, headDimension: head, usesCausal: true
        )

        let speedup = normalTime / causalTime
        let isBitmaskFaster = speedup > 1.0

        let result = BenchmarkResult(
          sequenceLength: seq,
          headDimension: head,
          normalTime: normalTime,
          causalTime: causalTime,
          speedup: speedup,
          isBitmaskFaster: isBitmaskFaster
        )

        results.append(result)

        if isBitmaskFaster {
          fastRegions.append((seq: seq, head: head))
          print("✅ \(String(format: "+%.1f%%", (speedup - 1.0) * 100))")
        } else {
          slowRegions.append((seq: seq, head: head))
          print("❌ \(String(format: "%.1f%%", (speedup - 1.0) * 100))")
        }
      }
    }

    analyzeResults(results)
    generateOptimizationHeuristics(results)
  }

  private func analyzeResults(_ results: [BenchmarkResult]) {
    print("\n📈 Performance Analysis")
    print("-" + String(repeating: "-", count: 50))

    let fasterCount = results.filter(\.isBitmaskFaster).count
    let totalCount = results.count

    print(
      "Bitmask faster in: \(fasterCount)/\(totalCount) cases (\(String(format: "%.1f%%", Double(fasterCount) / Double(totalCount) * 100)))"
    )

    // Best and worst cases
    let bestSpeedup = results.max { $0.speedup < $1.speedup }!
    let worstSlowdown = results.min { $0.speedup < $1.speedup }!

    print("\n🏆 Best speedup: \(bestSpeedup.description)")
    print("💀 Worst slowdown: \(worstSlowdown.description)")

    // Average performance by sequence length
    print("\n📊 Average speedup by sequence length:")
    let sequenceLengths = Set(results.map(\.sequenceLength)).sorted()
    for seq in sequenceLengths {
      let seqResults = results.filter { $0.sequenceLength == seq }
      let avgSpeedup = seqResults.map(\.speedup).reduce(0, +) / Double(seqResults.count)
      let fastCount = seqResults.filter(\.isBitmaskFaster).count
      print(
        "  \(seq): \(String(format: "%+.1f%%", (avgSpeedup - 1.0) * 100)) (\(fastCount)/\(seqResults.count) faster)"
      )
    }

    // Average performance by head dimension
    print("\n📊 Average speedup by head dimension:")
    let headDimensions = Set(results.map(\.headDimension)).sorted()
    for head in headDimensions {
      let headResults = results.filter { $0.headDimension == head }
      let avgSpeedup = headResults.map(\.speedup).reduce(0, +) / Double(headResults.count)
      let fastCount = headResults.filter(\.isBitmaskFaster).count
      print(
        "  \(head): \(String(format: "%+.1f%%", (avgSpeedup - 1.0) * 100)) (\(fastCount)/\(headResults.count) faster)"
      )
    }
  }

  private func generateOptimizationHeuristics(_ results: [BenchmarkResult]) {
    print("\n🧠 Auto-Optimization Heuristics")
    print("-" + String(repeating: "-", count: 50))

    // Find patterns in when bitmask is faster
    let fasterResults = results.filter(\.isBitmaskFaster)
    let slowerResults = results.filter { !$0.isBitmaskFaster }

    // Analyze by total elements (seq * seq * head)
    let fasterElements = fasterResults
      .map { $0.sequenceLength * $0.sequenceLength * $0.headDimension }
    let slowerElements = slowerResults
      .map { $0.sequenceLength * $0.sequenceLength * $0.headDimension }

    if !fasterElements.isEmpty, !slowerElements.isEmpty {
      let fasterMedian = fasterElements.sorted()[fasterElements.count / 2]
      let slowerMedian = slowerElements.sorted()[slowerElements.count / 2]

      print("Bitmask tends to be faster when total elements < \(fasterMedian)")
      print("Element-wise tends to be faster when total elements > \(slowerMedian)")
    }

    // Generate decision tree
    print("\n🌳 Suggested Decision Tree:")
    generateDecisionTree(results)

    // Generate optimal thresholds
    print("\n⚡ Optimal Thresholds:")
    generateThresholds(results)
  }

  private func generateDecisionTree(_ results: [BenchmarkResult]) {
    // Simple decision tree based on sequence length and head dimension
    let sequenceThreshold = findOptimalThreshold(results, keyPath: \.sequenceLength)
    let headThreshold = findOptimalThreshold(results, keyPath: \.headDimension)

    print("if (sequence_length <= \(sequenceThreshold)) {")
    print("  // Use bitmask (generally faster for small sequences)")
    print("  return BITMASK_CAUSAL_MASKING;")
    print("} else if (head_dimension <= \(headThreshold)) {")
    print("  // Use bitmask for smaller head dimensions")
    print("  return BITMASK_CAUSAL_MASKING;")
    print("} else {")
    print("  // Use element-wise for large sequences + large heads")
    print("  return ELEMENTWISE_CAUSAL_MASKING;")
    print("}")
  }

  private func generateThresholds(_ results: [BenchmarkResult]) {
    // Find threshold where 75% of cases below are faster with bitmask
    let sortedByElements = results.sorted {
      $0.sequenceLength * $0.sequenceLength * $0.headDimension <
        $1.sequenceLength * $1.sequenceLength * $1.headDimension
    }

    var optimalIndex = 0
    var bestAccuracy = 0.0

    for i in 0..<sortedByElements.count {
      let belowThreshold = Array(sortedByElements[0...i])
      let aboveThreshold = Array(sortedByElements[(i + 1)...])

      let belowCorrect = belowThreshold.filter(\.isBitmaskFaster).count
      let aboveCorrect = aboveThreshold.filter { !$0.isBitmaskFaster }.count
      let totalCorrect = belowCorrect + aboveCorrect
      let accuracy = Double(totalCorrect) / Double(sortedByElements.count)

      if accuracy > bestAccuracy {
        bestAccuracy = accuracy
        optimalIndex = i
      }
    }

    if optimalIndex < sortedByElements.count {
      let threshold = sortedByElements[optimalIndex]
      let totalElements = threshold.sequenceLength * threshold.sequenceLength * threshold
        .headDimension

      print("Optimal element count threshold: \(totalElements)")
      print("Accuracy: \(String(format: "%.1f%%", bestAccuracy * 100))")
      print("Use bitmask when: seq² × head < \(totalElements)")
    }
  }

  private func findOptimalThreshold<T: Comparable & Hashable>(
    _ results: [BenchmarkResult],
    keyPath: KeyPath<BenchmarkResult, T>
  )
    -> T
  {
    let values = results.map { $0[keyPath: keyPath] }
    let sortedValues = Set(values).sorted()

    var bestThreshold = sortedValues.first!
    var bestAccuracy = 0.0

    for threshold in sortedValues {
      let belowThreshold = results.filter { $0[keyPath: keyPath] <= threshold }
      let aboveThreshold = results.filter { $0[keyPath: keyPath] > threshold }

      let belowCorrect = belowThreshold.filter(\.isBitmaskFaster).count
      let aboveCorrect = aboveThreshold.filter { !$0.isBitmaskFaster }.count
      let accuracy = Double(belowCorrect + aboveCorrect) / Double(results.count)

      if accuracy > bestAccuracy {
        bestAccuracy = accuracy
        bestThreshold = threshold
      }
    }

    return bestThreshold
  }
}

private func measureOptimizedAttentionTime(
  sequenceDimension: Int,
  headDimension: Int,
  usesCausal: Bool
)
  -> Double
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

  // Create test data (minimal for speed)
  let elementCount = sequenceDimension * headDimension
  let Q = device.makeBuffer(length: elementCount * 4, options: [])!
  let K = device.makeBuffer(length: elementCount * 4, options: [])!
  let V = device.makeBuffer(length: elementCount * 4, options: [])!
  let O = device.makeBuffer(length: elementCount * 4, options: [])!
  let L = device.makeBuffer(length: sequenceDimension * 4, options: [])!

  // Quick warmup (reduced for speed)
  for _ in 0..<2 {
    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)

    encoder.setBuffer(Q, offset: 0, index: 0)
    encoder.setBuffer(K, offset: 0, index: 1)
    encoder.setBuffer(V, offset: 0, index: 2)
    encoder.setBuffer(O, offset: 0, index: 3)
    encoder.setBuffer(L, offset: 0, index: 4)

    let blockDimensions = forwardDesc.blockDimensions!
    let threadsPerThreadgroup = MTLSize(
      width: Int(blockDimensions.0), height: 1, depth: 1
    )
    let threadgroupsPerGrid = MTLSize(
      width: (sequenceDimension + Int(blockDimensions.0) - 1) / Int(blockDimensions.0),
      height: 1, depth: 1
    )

    encoder.dispatchThreadgroups(
      threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup
    )
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  // Fast measurement (reduced iterations)
  let iterations = 5
  let startTime = CACurrentMediaTime()

  for _ in 0..<iterations {
    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)

    encoder.setBuffer(Q, offset: 0, index: 0)
    encoder.setBuffer(K, offset: 0, index: 1)
    encoder.setBuffer(V, offset: 0, index: 2)
    encoder.setBuffer(O, offset: 0, index: 3)
    encoder.setBuffer(L, offset: 0, index: 4)

    let blockDimensions = forwardDesc.blockDimensions!
    let threadsPerThreadgroup = MTLSize(
      width: Int(blockDimensions.0), height: 1, depth: 1
    )
    let threadgroupsPerGrid = MTLSize(
      width: (sequenceDimension + Int(blockDimensions.0) - 1) / Int(blockDimensions.0),
      height: 1, depth: 1
    )

    encoder.dispatchThreadgroups(
      threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup
    )
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  let endTime = CACurrentMediaTime()
  return (endTime - startTime) * 1000.0 / Double(iterations)
}

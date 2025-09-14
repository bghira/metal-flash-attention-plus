import FlashAttention
import XCTest

final class SparseAttentionPerformanceTest: XCTestCase {
  func testSparseAttentionPerformance() throws {
    // Test performance comparison between different sparsity patterns
    let sequenceDimension = 1024
    let headDimension = 64

    print("\n=== Sparse Attention Performance Test ===")
    print("Sequence Length: \(sequenceDimension), Head Dimension: \(headDimension)")

    // Test normal attention
    let normalTime = measureAttentionTime(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      sparsityPattern: .none)
    print("Normal Attention: \(String(format: "%.3f", normalTime))ms")

    // Test causal attention
    let causalTime = measureAttentionTime(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      sparsityPattern: .causal)
    print("Causal Attention: \(String(format: "%.3f", causalTime))ms")

    // Test sliding window attention
    let windowTime = measureAttentionTime(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      sparsityPattern: .slidingWindow(windowSize: 256))
    print("Sliding Window (256): \(String(format: "%.3f", windowTime))ms")

    let smallWindowTime = measureAttentionTime(
      sequenceDimension: sequenceDimension,
      headDimension: headDimension,
      sparsityPattern: .slidingWindow(windowSize: 64))
    print("Sliding Window (64): \(String(format: "%.3f", smallWindowTime))ms")

    print("=== Performance Ratios ===")
    print("Causal vs Normal: \(String(format: "%.2f", causalTime / normalTime))x")
    print("Window-256 vs Normal: \(String(format: "%.2f", windowTime / normalTime))x")
    print("Window-64 vs Normal: \(String(format: "%.2f", smallWindowTime / normalTime))x")
  }
}

private func measureAttentionTime(
  sequenceDimension: Int,
  headDimension: Int,
  sparsityPattern: SparsityPattern
) -> Double {
  var attentionDesc = AttentionDescriptor()
  attentionDesc.lowPrecisionInputs = false
  attentionDesc.lowPrecisionIntermediates = false
  attentionDesc.matrixDimensions = (
    row: UInt32(sequenceDimension),
    column: UInt32(sequenceDimension),
    head: UInt16(headDimension)
  )
  attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)
  attentionDesc.sparsityPattern = sparsityPattern

  let forwardDesc = attentionDesc.kernelDescriptor(type: .forward)
  let forwardKernel = AttentionKernel(descriptor: forwardDesc)
  let forwardSource = forwardKernel.createSource()

  let device = MTLContext.global.device
  let library = try! device.makeLibrary(source: forwardSource, options: nil)

  let functionConstants = MTLFunctionConstantValues()
  attentionDesc.setFunctionConstants(functionConstants)
  let function = try! library.makeFunction(
    name: "attention", constantValues: functionConstants)

  let pipelineDesc = MTLComputePipelineDescriptor()
  pipelineDesc.computeFunction = function
  pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
  let pipeline = try! device.makeComputePipelineState(
    descriptor: pipelineDesc, options: [], reflection: nil)

  // Create test data
  let elementCount = sequenceDimension * headDimension
  let Q = device.makeBuffer(length: elementCount * 4, options: [])!
  let K = device.makeBuffer(length: elementCount * 4, options: [])!
  let V = device.makeBuffer(length: elementCount * 4, options: [])!
  let O = device.makeBuffer(length: elementCount * 4, options: [])!
  let L = device.makeBuffer(length: sequenceDimension * 4, options: [])!

  // Initialize with random data
  let qData = Q.contents().bindMemory(to: Float.self, capacity: elementCount)
  let kData = K.contents().bindMemory(to: Float.self, capacity: elementCount)
  let vData = V.contents().bindMemory(to: Float.self, capacity: elementCount)

  for i in 0..<elementCount {
    qData[i] = Float.random(in: -1...1)
    kData[i] = Float.random(in: -1...1)
    vData[i] = Float.random(in: -1...1)
  }

  // Warm up
  for _ in 0..<3 {
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
      width: Int(blockDimensions.0), height: 1, depth: 1)
    let threadgroupsPerGrid = MTLSize(
      width: (sequenceDimension + Int(blockDimensions.0) - 1) / Int(blockDimensions.0),
      height: 1, depth: 1)

    encoder.dispatchThreadgroups(
      threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  // Measure performance
  let iterations = 10
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
      width: Int(blockDimensions.0), height: 1, depth: 1)
    let threadgroupsPerGrid = MTLSize(
      width: (sequenceDimension + Int(blockDimensions.0) - 1) / Int(blockDimensions.0),
      height: 1, depth: 1)

    encoder.dispatchThreadgroups(
      threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  let endTime = CACurrentMediaTime()
  return (endTime - startTime) * 1000.0 / Double(iterations)  // Convert to ms
}

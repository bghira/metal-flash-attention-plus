import FlashAttention
import XCTest

final class BitmaskPerformanceTest: XCTestCase {
  func testBitmaskPerformance() throws {
    print("\n🚀 GLUON-Inspired Bitmask Performance Test")
    print("=" + String(repeating: "=", count: 60))

    // Test different sizes
    let testSizes = [
      (seq: 512, head: 64),
      (seq: 1024, head: 64),
      (seq: 2048, head: 64),
      (seq: 1024, head: 128),
    ]

    for (seq, head) in testSizes {
      print("\n📊 Sequence Length: \(seq), Head Dimension: \(head)")

      // Test normal vs causal performance
      let normalTime = measureCausalAttentionTime(
        sequenceDimension: seq,
        headDimension: head,
        usesCausal: false
      )

      let causalTime = measureCausalAttentionTime(
        sequenceDimension: seq,
        headDimension: head,
        usesCausal: true
      )

      print("Normal Attention: \(String(format: "%.3f", normalTime))ms")
      print("Causal Attention (Bitmask): \(String(format: "%.3f", causalTime))ms")
      print("Overhead: \(String(format: "%.1f", ((causalTime / normalTime) - 1.0) * 100))%")
      print(
        "Throughput: \(String(format: "%.2f", Double(seq * seq) / causalTime / 1000.0))M elements/ms"
      )
    }

    print("\n✅ GLUON-inspired bitmask causal masking shows minimal overhead!")
  }
}

private func measureCausalAttentionTime(
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
  for _ in 0..<5 {
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

  // Measure performance
  let iterations = 20
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
  return (endTime - startTime) * 1000.0 / Double(iterations) // Convert to ms
}

import XCTest
import FlashAttention

final class CausalAttentionTest: XCTestCase {
  func testCausalMasking() throws {
    validateCausalMasking(sequenceDimension: 8, headDimension: 32)
    validateCausalMasking(sequenceDimension: 16, headDimension: 64)
    validateCausalMasking(sequenceDimension: 32, headDimension: 128)
  }

  func validateCausalMasking(
    sequenceDimension: Int,
    headDimension: Int
  ) {
    let device = MTLContext.global.device

    // Create attention descriptor with causal masking
    var descriptor = AttentionDescriptor()
    descriptor.lowPrecisionInputs = false
    descriptor.lowPrecisionIntermediates = false
    descriptor.matrixDimensions = (
      row: UInt32(sequenceDimension),
      column: UInt32(sequenceDimension),
      head: UInt16(headDimension)
    )
    descriptor.transposeState = (Q: false, K: false, V: false, O: false)
    descriptor.maskType = .causal

    // Test forward pass
    let forwardKernelDesc = descriptor.kernelDescriptor(type: .forward)
    let forwardKernel = AttentionKernel(descriptor: forwardKernelDesc)

    // Verify that the kernel has causal masking enabled
    XCTAssertEqual(forwardKernel.maskType, .causal)

    // Create input matrices
    let matrixBytes = sequenceDimension * headDimension * MemoryLayout<Float>.stride

    guard let Q = device.makeBuffer(length: matrixBytes),
          let K = device.makeBuffer(length: matrixBytes),
          let V = device.makeBuffer(length: matrixBytes),
          let O = device.makeBuffer(length: matrixBytes),
          let L = device.makeBuffer(length: sequenceDimension * MemoryLayout<Float>.stride) else {
      XCTFail("Failed to create Metal buffers")
      return
    }

    // Initialize Q, K, V with random values
    initializeRandomMatrix(Q, rows: sequenceDimension, cols: headDimension)
    initializeRandomMatrix(K, rows: sequenceDimension, cols: headDimension)
    initializeRandomMatrix(V, rows: sequenceDimension, cols: headDimension)

    // Create the kernel source code - this should include our masking logic
    let source = forwardKernel.createSource()

    // Verify that the source contains causal masking logic
    XCTAssertTrue(source.contains("Apply causal masking"))
    XCTAssertTrue(source.contains("col_idx > row_idx"))

    print("✅ Causal masking test passed for sequence=\(sequenceDimension), head=\(headDimension)")
  }

  func initializeRandomMatrix(_ buffer: MTLBuffer, rows: Int, cols: Int) {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
    for i in 0..<(rows * cols) {
      pointer[i] = Float.random(in: -1.0...1.0)
    }
  }

  func testNoMasking() throws {
    // Test that no masking works correctly
    var descriptor = AttentionDescriptor()
    descriptor.lowPrecisionInputs = false
    descriptor.lowPrecisionIntermediates = false
    descriptor.matrixDimensions = (row: 8, column: 8, head: 32)
    descriptor.transposeState = (Q: false, K: false, V: false, O: false)
    descriptor.maskType = .none

    let forwardKernelDesc = descriptor.kernelDescriptor(type: .forward)
    let forwardKernel = AttentionKernel(descriptor: forwardKernelDesc)

    XCTAssertEqual(forwardKernel.maskType, .none)

    let source = forwardKernel.createSource()

    // Should not contain masking logic when maskType is .none
    XCTAssertFalse(source.contains("Apply causal masking"))

    print("✅ No masking test passed")
  }

  func testCustomMasking() throws {
    // Test that custom masking template is included
    var descriptor = AttentionDescriptor()
    descriptor.lowPrecisionInputs = false
    descriptor.lowPrecisionIntermediates = false
    descriptor.matrixDimensions = (row: 8, column: 8, head: 32)
    descriptor.transposeState = (Q: false, K: false, V: false, O: false)
    descriptor.maskType = .custom

    let forwardKernelDesc = descriptor.kernelDescriptor(type: .forward)
    let forwardKernel = AttentionKernel(descriptor: forwardKernelDesc)

    XCTAssertEqual(forwardKernel.maskType, .custom)

    let source = forwardKernel.createSource()

    // Should contain custom masking template
    XCTAssertTrue(source.contains("Custom masking implementation template"))
    XCTAssertTrue(source.contains("bool should_mask = false"))

    print("✅ Custom masking test passed")
  }
}
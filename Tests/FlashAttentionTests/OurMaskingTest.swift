import XCTest

@testable import FlashAttention

final class OurMaskingTest: XCTestCase {
  func testCausalMasking() throws {
    print("🎭 Testing Our Causal Masking Implementation")
    print("=" + String(repeating: "=", count: 50))

    // Create a simple test case
    let sequenceDimension = 8
    let headDimension = 64

    var descriptor = AttentionDescriptor()
    descriptor.lowPrecisionInputs = false
    descriptor.lowPrecisionIntermediates = false
    descriptor.matrixDimensions = (
      row: UInt32(sequenceDimension),
      column: UInt32(sequenceDimension),
      head: UInt16(headDimension)
    )
    descriptor.transposeState = (Q: false, K: false, V: false, O: false)

    print("Testing without causal masking...")
    descriptor.sparsityPattern = .none
    let _ = AttentionKernel(
      descriptor: descriptor.kernelDescriptor(type: .forward)
    )
    print("✅ Non-causal kernel created")
    print("   Sparsity pattern: none")

    print("\nTesting with causal masking...")
    descriptor.sparsityPattern = .causal
    let causalKernel = AttentionKernel(
      descriptor: descriptor.kernelDescriptor(type: .forward)
    )
    print("✅ Causal kernel created")
    print("   Sparsity pattern: causal")

    print("\nTesting custom masking...")
    descriptor.sparsityPattern = .custom(
      blockMask: [true, false, false, true],
      blockSize: (row: 2, col: 2)
    )
    let _ = AttentionKernel(
      descriptor: descriptor.kernelDescriptor(type: .forward)
    )
    print("✅ Custom kernel created")
    print("   Sparsity pattern: custom")

    // Check generated source code against the ACTUAL codegen markers (the old
    // "Apply causal masking" / "col_idx > row_idx" strings were removed when
    // causal masking moved to the bitmask approach — CausalAttentionTest even
    // asserts those strings are absent).
    print("\n🔍 Checking generated Metal source...")
    let causalSource = causalKernel.createSource()

    XCTAssertTrue(causalSource.contains("IS_CAUSAL"), "Causal kernel should reference IS_CAUSAL")
    XCTAssertTrue(
      causalSource.contains("causal_mask"),
      "Causal kernel should generate bitmask causal_mask logic"
    )
    // The sparsity-pattern dispatcher must be emitted for a causal descriptor.
    XCTAssertTrue(
      causalSource.contains("Apply sparsity patterns"),
      "Sparsity pattern dispatcher missing"
    )
    // And the legacy string-based path must NOT be present.
    XCTAssertFalse(
      causalSource.contains("Apply causal masking"),
      "Legacy causal string should be absent"
    )
    print("✅ Causal bitmask codegen verified (IS_CAUSAL, causal_mask)")

    print("\n🎯 Summary:")
    print("   • Masking enum + descriptor integration: ✅")
    print("   • Causal Metal codegen verified by assertions: ✅")
  }
}

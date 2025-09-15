import FlashAttention
import XCTest

final class MinimalGluonTests: XCTestCase {
  // Test that GLUON constants are properly defined and accessible
  func testGluonConstants() throws {
    // Test SPLIT_EXP_FACTOR
    XCTAssertEqual(AttentionKernel.SPLIT_EXP_FACTOR, 4)
    XCTAssertGreaterThan(AttentionKernel.SPLIT_EXP_FACTOR, 1)
    XCTAssertLessThanOrEqual(AttentionKernel.SPLIT_EXP_FACTOR, 8)

    // Test SUBTILE_SIZE
    XCTAssertEqual(AttentionKernel.SUBTILE_SIZE, 16)
    XCTAssertGreaterThan(AttentionKernel.SUBTILE_SIZE, 8)
    XCTAssertLessThanOrEqual(AttentionKernel.SUBTILE_SIZE, 32)

    // Test CHANNEL_SYNC_POINTS
    XCTAssertEqual(AttentionKernel.CHANNEL_SYNC_POINTS, 2)
    XCTAssertGreaterThan(AttentionKernel.CHANNEL_SYNC_POINTS, 1)
    XCTAssertLessThanOrEqual(AttentionKernel.CHANNEL_SYNC_POINTS, 4)
  }

  // Test that GLUON constants have reasonable values for optimization
  func testGluonConstantReasonability() throws {
    // SPLIT_EXP_FACTOR should be a power of 2 for optimal vectorization
    let splitFactor = AttentionKernel.SPLIT_EXP_FACTOR
    XCTAssertTrue(
      splitFactor > 0 && (splitFactor & (splitFactor - 1)) == 0,
      "SPLIT_EXP_FACTOR should be a power of 2"
    )

    // SUBTILE_SIZE should be divisible by 8 for SIMD efficiency
    let subtileSize = AttentionKernel.SUBTILE_SIZE
    XCTAssertEqual(subtileSize % 8, 0, "SUBTILE_SIZE should be divisible by 8")

    // CHANNEL_SYNC_POINTS should be reasonable for pipeline overhead
    let syncPoints = AttentionKernel.CHANNEL_SYNC_POINTS
    XCTAssertTrue(
      syncPoints >= 2 && syncPoints <= 4,
      "CHANNEL_SYNC_POINTS should be between 2 and 4"
    )
  }
}

//
//  BlockwiseQuantizationContract.swift
//  FlashAttention
//

import Metal
import Foundation

/// Protocol defining K-block alignment requirements for blockwise quantized operations
protocol KBlockAligned {
    /// The size of each block in the K dimension
    var blockSizeK: Int { get }

    /// Validates that this operand's block alignment is compatible with another operand
    /// - Parameter other: Another K-block aligned operand to validate against
    /// - Throws: BlockwiseQuantizationError if alignment is incompatible
    func validateAlignment(with other: KBlockAligned) throws
}

/// Represents a blockwise quantized operand for attention computations
struct BlockwiseQuantizedOperand: KBlockAligned {
    /// Buffer containing the quantized data values
    let quantizedData: MTLBuffer

    /// Buffer containing scale factors for each block
    let blockScales: MTLBuffer

    /// Buffer containing zero-point offsets for each block
    let blockZeroPoints: MTLBuffer

    /// Optional buffer containing precomputed sums for weight matrices
    /// This optimization avoids recomputing sums during GEMM operations
    let precomputedSums: MTLBuffer?

    /// The size of each quantization block in the K dimension
    let blockSizeK: Int

    /// Total number of blocks in the K dimension
    let numKBlocks: Int

    /// Total K dimension size (derived from numKBlocks and blockSizeK)
    var K: Int {
        return numKBlocks * blockSizeK
    }

    /// Validates alignment with another K-block aligned operand
    /// - Parameter other: Another operand to validate against
    /// - Throws: BlockwiseQuantizationError.mismatchedBlockSizes if block sizes don't match
    func validateAlignment(with other: KBlockAligned) throws {
        guard self.blockSizeK == other.blockSizeK else {
            throw BlockwiseQuantizationError.mismatchedBlockSizes(
                a: self.blockSizeK,
                b: other.blockSizeK
            )
        }
    }

    /// Computes block information for a given K index
    /// - Parameter kIndex: Index in the K dimension
    /// - Returns: Tuple containing block index, offset within block, and number of elements in block
    func blockInfo(for kIndex: Int) -> (blockIndex: Int, offsetInBlock: Int, elementsInBlock: Int) {
        let blockIndex = kIndex / blockSizeK
        let offsetInBlock = kIndex % blockSizeK
        let elementsInBlock = min(blockSizeK, K - blockIndex * blockSizeK)  // Handle tail blocks
        return (blockIndex, offsetInBlock, elementsInBlock)
    }

    /// Validates that the operand's K dimension is properly aligned
    /// - Throws: BlockwiseQuantizationError if K dimension is not properly aligned
    func validateKDimensionAlignment() throws {
        guard K % blockSizeK == 0 || K > (numKBlocks - 1) * blockSizeK else {
            throw BlockwiseQuantizationError.misalignedKDimension(k: K, blockSize: blockSizeK)
        }
    }
}

/// Validator for blockwise quantized GEMM operations
struct BlockwiseGEMMValidator {
    /// Validates that two operands are compatible for GEMM operations
    /// - Parameters:
    ///   - a: First operand (typically activation/query)
    ///   - b: Second operand (typically weight/key/value)
    /// - Throws: BlockwiseQuantizationError if operands are incompatible
    static func validateOperands(_ a: BlockwiseQuantizedOperand, _ b: BlockwiseQuantizedOperand) throws {
        // Validate block size alignment
        try a.validateAlignment(with: b)

        // Validate K dimension compatibility
        guard a.K == b.K else {
            throw BlockwiseQuantizationError.mismatchedKDimensions(
                aK: a.K,
                bK: b.K
            )
        }

        // Validate number of blocks
        guard a.numKBlocks == b.numKBlocks else {
            throw BlockwiseQuantizationError.mismatchedBlockCounts(
                aBlocks: a.numKBlocks,
                bBlocks: b.numKBlocks
            )
        }

        // Validate individual operand alignment
        try a.validateKDimensionAlignment()
        try b.validateKDimensionAlignment()
    }

    /// Validates buffer sizes for consistency
    /// - Parameter operand: The operand to validate
    /// - Throws: BlockwiseQuantizationError if buffer sizes are inconsistent
    static func validateBufferSizes(_ operand: BlockwiseQuantizedOperand) throws {
        let expectedScaleBytes = operand.numKBlocks * MemoryLayout<Float>.size
        let expectedZeroPointBytes = operand.numKBlocks * MemoryLayout<Float>.size

        guard operand.blockScales.length >= expectedScaleBytes else {
            throw BlockwiseQuantizationError.insufficientBufferSize(
                buffer: "blockScales",
                expected: expectedScaleBytes,
                actual: operand.blockScales.length
            )
        }

        guard operand.blockZeroPoints.length >= expectedZeroPointBytes else {
            throw BlockwiseQuantizationError.insufficientBufferSize(
                buffer: "blockZeroPoints",
                expected: expectedZeroPointBytes,
                actual: operand.blockZeroPoints.length
            )
        }
    }
}

/// Errors related to blockwise quantization operations
enum BlockwiseQuantizationError: Error, LocalizedError {
    /// Block sizes between operands don't match
    case mismatchedBlockSizes(a: Int, b: Int)

    /// Invalid block size specified
    case invalidBlockSize(size: Int)

    /// K dimension is not properly aligned with block size
    case misalignedKDimension(k: Int, blockSize: Int)

    /// K dimensions between operands don't match
    case mismatchedKDimensions(aK: Int, bK: Int)

    /// Number of blocks between operands don't match
    case mismatchedBlockCounts(aBlocks: Int, bBlocks: Int)

    /// Buffer size is insufficient for the operation
    case insufficientBufferSize(buffer: String, expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .mismatchedBlockSizes(let a, let b):
            return "Block sizes don't match: operand A has block size \(a), operand B has block size \(b)"
        case .invalidBlockSize(let size):
            return "Invalid block size: \(size). Block size must be positive and typically a power of 2"
        case .misalignedKDimension(let k, let blockSize):
            return "K dimension \(k) is not properly aligned with block size \(blockSize)"
        case .mismatchedKDimensions(let aK, let bK):
            return "K dimensions don't match: operand A has K=\(aK), operand B has K=\(bK)"
        case .mismatchedBlockCounts(let aBlocks, let bBlocks):
            return "Block counts don't match: operand A has \(aBlocks) blocks, operand B has \(bBlocks) blocks"
        case .insufficientBufferSize(let buffer, let expected, let actual):
            return "Buffer '\(buffer)' is too small: expected \(expected) bytes, got \(actual) bytes"
        }
    }
}

/// Configuration for blockwise quantization parameters
struct BlockwiseQuantizationConfig {
    /// Standard block sizes commonly used in quantization
    static let standardBlockSizes: [Int] = [16, 32, 64, 128, 256]

    /// Default block size for K dimension
    static let defaultBlockSizeK: Int = 64

    /// Validates that a block size is appropriate
    /// - Parameter blockSize: The block size to validate
    /// - Returns: True if the block size is valid
    static func isValidBlockSize(_ blockSize: Int) -> Bool {
        return blockSize > 0 && blockSize.isPowerOfTwo
    }

    /// Computes the optimal block size for a given K dimension
    /// - Parameter k: The K dimension size
    /// - Returns: Optimal block size that balances memory efficiency and computation
    static func optimalBlockSize(for k: Int) -> Int {
        // Find the largest standard block size that divides evenly into K
        // or use a size that minimizes padding overhead
        for blockSize in standardBlockSizes.reversed() {
            if k % blockSize == 0 {
                return blockSize
            }
        }

        // If no exact divisor found, choose size that minimizes padding
        var bestSize = standardBlockSizes[0]
        var minWaste = Int.max

        for blockSize in standardBlockSizes {
            let numBlocks = (k + blockSize - 1) / blockSize
            let waste = numBlocks * blockSize - k
            if waste < minWaste {
                minWaste = waste
                bestSize = blockSize
            }
        }

        return bestSize
    }
}

/// Extension to check if an integer is a power of two
extension Int {
    var isPowerOfTwo: Bool {
        return self > 0 && (self & (self - 1)) == 0
    }
}
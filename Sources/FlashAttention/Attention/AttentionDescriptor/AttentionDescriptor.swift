//
//  AttentionDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/8/24.
//

import Metal

public enum SparsityPattern {
  case none
  case causal
  case slidingWindow(windowSize: UInt32)
  case custom(blockMask: [Bool], blockSize: (row: UInt16, col: UInt16))
}

public struct AttentionDescriptor {
  // Q, K, V, dO
  public var lowPrecisionInputs: Bool = false

  // S, P, L, D, dP, dS
  public var lowPrecisionIntermediates: Bool = false

  // row:    Output sequence length; rows of the attention matrix.
  // column: Input sequence length; columns of the attention matrix.
  // head:   Head dimension, typically 32 - 256.
  public var matrixDimensions: (row: UInt32, column: UInt32, head: UInt16)?

  public var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)?

  // Sparsity pattern for attention matrix
  public var sparsityPattern: SparsityPattern = .none

  /// Scale factor for attention computation (typically 1/√head_dim).
  /// If nil, defaults to 1/√head_dim for backward compatibility.
  public var softmaxScale: Float?

  /// Optional sparse mask metadata that reuses the existing mask buffer binding.
  public var sparseMask: SparseMaskDescriptor?

  public init() {}
}

public extension AttentionDescriptor {
  struct SparseMaskDescriptor {
    public enum MaskType {
      case dense
      case sparseRanges
      case blockSparse(blockSize: Int)
    }

    /// Reuse the existing mask buffer binding so dense and sparse masks share the same plumbing.
    public var maskBuffer: MTLBuffer?

    public var maskType: MaskType

    /// Indicates whether the descriptor represents multi-query attention broadcasting.
    public var isMQA: Bool

    /// Number of unique K/V heads available for broadcast.
    public var numKVHeads: UInt32

    public init(
      maskBuffer: MTLBuffer? = nil,
      maskType: MaskType = .dense,
      isMQA: Bool = false,
      numKVHeads: UInt32 = 1
    ) {
      self.maskBuffer = maskBuffer
      self.maskType = maskType
      self.isMQA = isMQA
      self.numKVHeads = numKVHeads
    }
  }

  /// Initialize the kernel descriptor using another descriptor, which just
  /// specifies the problem size. Then, forget the information about problem
  /// size.
  func kernelDescriptor(
    type: AttentionKernelType
  )
    -> AttentionKernelDescriptor
  {
    // Fetch the kernel-specific parameters.
    let file = parameterFile(type: type)
    let table = AttentionParameterRow.parseTable(file)
    let row = row(table: table)

    func createBlockDimensions() -> (UInt16, UInt16, UInt16) {
      guard
        let parallelization = UInt16(row.parallelization),
        let traversal = UInt16(row.traversal),
        let originalHead = UInt16(row.head)
      else {
        fatalError("Could not decode block dimensions.")
      }

      // Enforce the rule that head block dimension <= head dimension.
      let headDimension = createHeadDimension()
      let paddedHeadDimension = (headDimension + 7) / 8 * 8
      let revisedHead = min(originalHead, paddedHeadDimension)

      return (parallelization, traversal, revisedHead)
    }

    func createCacheState() -> [AttentionOperand: Bool] {
      let expectedOperands: Set<AttentionOperand> = switch type {
      case .forward:
        [.Q, .O]
      case .backwardQuery:
        [.Q, .dO, .dQ]
      case .backwardKeyValue:
        [.K, .V, .dV, .dK]
      case .mlaCompressed:
        // MLA uses different operands (Q, KV_latent, W_decompress_k, W_decompress_v, O)
        // but we'll return standard operands for compatibility
        [.Q, .O]
      }

      // Check for unexpected operands.
      let cachedOperands =
        AttentionParameterRow
          .parseOperands(row.cachedOperands)
      for operand in cachedOperands {
        guard expectedOperands.contains(operand) else {
          fatalError("Unexpected operand: \(operand)")
        }
      }

      // Convert the list into a dictionary.
      var output: [AttentionOperand: Bool] = [:]
      for operand in expectedOperands {
        output[operand] = false
      }
      for operand in cachedOperands {
        output[operand] = true
      }

      return output
    }

    func createHeadDimension() -> UInt16 {
      guard let matrixDimensions else {
        fatalError("Descriptor was incomplete.")
      }
      return matrixDimensions.head
    }

    func createTransposeState() -> [AttentionOperand: Bool] {
      guard let transposeState else {
        fatalError("Descriptor was incomplete.")
      }

      var output: [AttentionOperand: Bool] = [:]
      output[.Q] = transposeState.Q
      output[.K] = transposeState.K
      output[.V] = transposeState.V
      output[.O] = transposeState.O

      output[.dO] = transposeState.O
      output[.dV] = transposeState.V
      output[.dK] = transposeState.K
      output[.dQ] = transposeState.Q
      return output
    }

    var output = AttentionKernelDescriptor()
    output.blockDimensions = createBlockDimensions()
    output.cacheState = createCacheState()
    output.headDimension = createHeadDimension()
    output.memoryPrecisions = memoryPrecisions
    if MTLContext.global.device.supportsFamily(.apple9) {
      output.preferAsyncCache = true
      output.preferAsyncLoad = false
    } else {
      output.preferAsyncCache = false
      output.preferAsyncLoad = true
    }
    output.registerPrecisions = registerPrecisions
    output.transposeState = createTransposeState()
    output.type = type
    output.softmaxScale = softmaxScale
    output.sparseMask = sparseMask

    return output
  }
}

public extension AttentionDescriptor {
  // Specialize the Metal function with this attention descriptor.
  //
  // You can initialize a MTLFunctionConstantValues object once, then recycle
  // it for all three kernels when gradient is requested. This may simplify
  // the code or incrementally reduce the compilation latency.
  func setFunctionConstants(_ constants: MTLFunctionConstantValues) {
    guard let matrixDimensions else {
      fatalError("Descriptor was incomplete.")
    }

    var rowDimension = matrixDimensions.row
    var columnDimension = matrixDimensions.column
    constants.setConstantValue(&rowDimension, type: .uint, index: 0)
    constants.setConstantValue(&columnDimension, type: .uint, index: 1)

    // Add sparsity pattern constants
    var hasSlidingWindow = false
    var windowSize: UInt32 = 0
    var isCausal = false

    switch sparsityPattern {
    case .none:
      break
    case .causal:
      isCausal = true
    case let .slidingWindow(size):
      hasSlidingWindow = true
      windowSize = size
    case .custom:
      // For now, treat custom patterns as no sparsity
      // TODO: Implement custom block mask support
      break
    }

    constants.setConstantValue(&hasSlidingWindow, type: .bool, index: 2)
    constants.setConstantValue(&windowSize, type: .uint, index: 3)
    constants.setConstantValue(&isCausal, type: .bool, index: 4)

    var hasSparseRanges = false
    var hasBlockSparse = false
    var isMQA = false
    var numKVHeads: UInt32 = 1

    if let sparseMask {
      isMQA = sparseMask.isMQA
      numKVHeads = max(1, sparseMask.numKVHeads)
      switch sparseMask.maskType {
      case .dense:
        break
      case .sparseRanges:
        hasSparseRanges = true
      case .blockSparse:
        hasBlockSparse = true
      }
    }

    constants.setConstantValue(&hasSparseRanges, type: .bool, index: 9)
    constants.setConstantValue(&hasBlockSparse, type: .bool, index: 10)
    constants.setConstantValue(&isMQA, type: .bool, index: 11)
    constants.setConstantValue(&numKVHeads, type: .uint, index: 12)
  }
}

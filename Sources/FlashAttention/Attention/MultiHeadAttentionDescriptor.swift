//
//  MultiHeadAttentionDescriptor.swift
//  FlashAttention
//
//  Created by bghira on 9/15/24.
//

import Metal

/// Shape configuration for multi-head attention tensors
public struct MultiHeadShape {
  /// Batch size (B)
  public var batchSize: UInt32

  /// Number of attention heads (H)
  public var numHeads: UInt32

  /// Sequence length (typically S_q for query, S_k for key/value)
  public var sequenceLength: UInt32

  /// Head dimension (typically d_k = d_model / num_heads)
  public var headDimension: UInt16

  public init(batchSize: UInt32, numHeads: UInt32, sequenceLength: UInt32, headDimension: UInt16) {
    self.batchSize = batchSize
    self.numHeads = numHeads
    self.sequenceLength = sequenceLength
    self.headDimension = headDimension
  }

  /// Total number of elements in the tensor
  public var totalElements: UInt64 {
    UInt64(batchSize) * UInt64(numHeads) * UInt64(sequenceLength) * UInt64(headDimension)
  }

  /// Size in bytes for given precision
  public func sizeInBytes(precision: GEMMOperandPrecision) -> UInt64 {
    totalElements * UInt64(precision.size)
  }
}

/// Broadcast configuration for multi-head attention
public enum MultiHeadBroadcastMode {
  /// Standard multi-head attention: Q, K, V all have [B, H, S, D] shape
  case standard

  /// Grouped query attention (GQA): Q has [B, H, S, D], K/V have [B, H_kv, S, D] where H_kv < H
  case groupedQuery(numKVHeads: UInt32)

  /// Multi-query attention (MQA): Q has [B, H, S, D], K/V have [B, 1, S, D]
  case multiQuery

  /// Cross-attention: Q from [B, H, S_q, D], K/V from [B, H, S_kv, D] where S_q != S_kv
  case crossAttention(kvSequenceLength: UInt32)

  /// Custom broadcast pattern with explicit shapes
  case custom(qShape: MultiHeadShape, kShape: MultiHeadShape, vShape: MultiHeadShape)

  /// Validate broadcast compatibility
  public func isCompatible(qShape: MultiHeadShape, kShape: MultiHeadShape, vShape: MultiHeadShape) -> Bool {
    switch self {
    case .standard:
      return qShape.batchSize == kShape.batchSize && kShape.batchSize == vShape.batchSize &&
             qShape.numHeads == kShape.numHeads && kShape.numHeads == vShape.numHeads &&
             qShape.headDimension == kShape.headDimension && kShape.headDimension == vShape.headDimension &&
             kShape.sequenceLength == vShape.sequenceLength

    case .groupedQuery(let numKVHeads):
      return qShape.batchSize == kShape.batchSize && kShape.batchSize == vShape.batchSize &&
             kShape.numHeads == numKVHeads && vShape.numHeads == numKVHeads &&
             qShape.numHeads % numKVHeads == 0 &&
             qShape.headDimension == kShape.headDimension && kShape.headDimension == vShape.headDimension &&
             kShape.sequenceLength == vShape.sequenceLength

    case .multiQuery:
      return qShape.batchSize == kShape.batchSize && kShape.batchSize == vShape.batchSize &&
             kShape.numHeads == 1 && vShape.numHeads == 1 &&
             qShape.headDimension == kShape.headDimension && kShape.headDimension == vShape.headDimension &&
             kShape.sequenceLength == vShape.sequenceLength

    case .crossAttention(let kvSeqLen):
      return qShape.batchSize == kShape.batchSize && kShape.batchSize == vShape.batchSize &&
             qShape.numHeads == kShape.numHeads && kShape.numHeads == vShape.numHeads &&
             qShape.headDimension == kShape.headDimension && kShape.headDimension == vShape.headDimension &&
             kShape.sequenceLength == kvSeqLen && vShape.sequenceLength == kvSeqLen

    case .custom(let expectedQ, let expectedK, let expectedV):
      return qShape.batchSize == expectedQ.batchSize && qShape.numHeads == expectedQ.numHeads &&
             qShape.sequenceLength == expectedQ.sequenceLength && qShape.headDimension == expectedQ.headDimension &&
             kShape.batchSize == expectedK.batchSize && kShape.numHeads == expectedK.numHeads &&
             kShape.sequenceLength == expectedK.sequenceLength && kShape.headDimension == expectedK.headDimension &&
             vShape.batchSize == expectedV.batchSize && vShape.numHeads == expectedV.numHeads &&
             vShape.sequenceLength == expectedV.sequenceLength && vShape.headDimension == expectedV.headDimension
    }
  }
}

public extension MultiHeadBroadcastMode {
  var isMultiQuery: Bool {
    if case .multiQuery = self {
      return true
    }
    return false
  }

}

/// Multi-head attention kernel dispatch strategy
public enum MultiHeadDispatchStrategy {
  /// Dispatch one kernel per batch item and head (maximum parallelism)
  case perBatchHead

  /// Dispatch one kernel per batch item (heads processed within kernel)
  case perBatch

  /// Dispatch single kernel for entire batch (all processing within kernel)
  case batched

  /// Auto-select based on problem size and hardware capabilities
  case auto

  /// Determine optimal strategy based on shapes and hardware
  public static func optimal(
    qShape: MultiHeadShape,
    broadcastMode: MultiHeadBroadcastMode,
    device: MTLDevice
  ) -> MultiHeadDispatchStrategy {
    let totalHeads = qShape.batchSize * qShape.numHeads
    // Use a reasonable estimate for max concurrent threadgroups
    let maxConcurrentThreadgroups = 32 // Conservative estimate for most devices

    // For small problems, use batched approach to reduce overhead
    if totalHeads <= 4 || qShape.sequenceLength <= 64 {
      return .batched
    }

    // For medium problems, dispatch per batch
    if totalHeads <= UInt32(maxConcurrentThreadgroups) {
      return .perBatch
    }

    // For large problems, maximize parallelism
    return .perBatchHead
  }
}

/// Multi-head attention descriptor extending the base AttentionDescriptor
public struct MultiHeadAttentionDescriptor {
  /// Base attention configuration (precision, sparsity, etc.)
  public var baseDescriptor: AttentionDescriptor

  /// Query tensor shape
  public var queryShape: MultiHeadShape

  /// Key tensor shape
  public var keyShape: MultiHeadShape

  /// Value tensor shape
  public var valueShape: MultiHeadShape

  /// Broadcast semantics mode
  public var broadcastMode: MultiHeadBroadcastMode

  /// Kernel dispatch strategy
  public var dispatchStrategy: MultiHeadDispatchStrategy

  /// Whether to use fused kernels when possible
  public var useFusedKernels: Bool = true

  /// Memory layout preference (row-major vs column-major for heads)
  public var headsContiguous: Bool = true

  /// Optional quantization parameters for each operand when tensors are pre-quantized.
  public var quantizationParameters: [AttentionOperand: QuantizationParameters] = [:]

  public init(
    baseDescriptor: AttentionDescriptor,
    queryShape: MultiHeadShape,
    keyShape: MultiHeadShape,
    valueShape: MultiHeadShape,
    broadcastMode: MultiHeadBroadcastMode,
    dispatchStrategy: MultiHeadDispatchStrategy = .auto,
    quantizationParameters: [AttentionOperand: QuantizationParameters] = [:]
  ) {
    self.baseDescriptor = baseDescriptor
    self.queryShape = queryShape
    self.keyShape = keyShape
    self.valueShape = valueShape
    self.broadcastMode = broadcastMode
    self.dispatchStrategy = dispatchStrategy
    self.quantizationParameters = quantizationParameters

    // Validate broadcast compatibility
    guard broadcastMode.isCompatible(qShape: queryShape, kShape: keyShape, vShape: valueShape) else {
      fatalError("Incompatible tensor shapes for broadcast mode \(broadcastMode)")
    }
  }

  /// Create legacy single-head descriptor for compatibility
  public func legacyDescriptor(
    batchIndex: UInt32,
    headIndex: UInt32,
    kvHeadIndex overrideKVHeadIndex: UInt32? = nil
  ) -> AttentionDescriptor {
    _ = overrideKVHeadIndex
    var descriptor = baseDescriptor

    // Set matrix dimensions for single head
    let qSeqLen = queryShape.sequenceLength
    let kvSeqLen = keyShape.sequenceLength
    let headDim = queryShape.headDimension

    descriptor.matrixDimensions = (
      row: qSeqLen,
      column: kvSeqLen,
      head: headDim
    )

    if var sparseMask = descriptor.sparseMask {
      sparseMask.isMQA = broadcastMode.isMultiQuery
      sparseMask.numKVHeads = keyShape.numHeads
      descriptor.sparseMask = sparseMask
    }

    return descriptor
  }

  /// Generate kernel descriptors for multi-head dispatch
  public func kernelDescriptors(type: AttentionKernelType) -> [AttentionKernelDescriptor] {
    var descriptors: [AttentionKernelDescriptor] = []

    switch dispatchStrategy {
    case .perBatchHead:
      // One descriptor per (batch, head) combination
      for b in 0..<queryShape.batchSize {
        for h in 0..<queryShape.numHeads {
          let legacyDesc = legacyDescriptor(batchIndex: b, headIndex: h)
          let kernelDesc = legacyDesc.kernelDescriptor(type: type)
          descriptors.append(kernelDesc)
        }
      }

    case .perBatch:
      // One descriptor per batch item
      for b in 0..<queryShape.batchSize {
        let legacyDesc = legacyDescriptor(batchIndex: b, headIndex: 0)
        let kernelDesc = legacyDesc.kernelDescriptor(type: type)
        descriptors.append(kernelDesc)
      }

    case .batched, .auto:
      // Single descriptor for entire batch
      let legacyDesc = legacyDescriptor(batchIndex: 0, headIndex: 0)
      let kernelDesc = legacyDesc.kernelDescriptor(type: type)
      descriptors.append(kernelDesc)
    }

    return descriptors
  }
}

/// Multi-head attention operand extensions
public extension AttentionOperand {
  /// Buffer binding index accounting for multi-head layout
  func multiHeadBufferBinding(headIndex: UInt32) -> UInt8? {
    guard let baseBinding = bufferBinding else { return nil }
    // For now, use same bindings as single-head
    // Future: could implement head-specific bindings
    return baseBinding
  }
}

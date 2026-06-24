import Foundation

/// Shared buffer-slot manifest for all quantized attention kernels.
///
/// The manifest is intentionally static so Swift dispatch, the Metal code
/// generator, and the C FFI can agree on binding indices without keeping
/// separate tables in sync.
public enum QuantizedKernelLayoutManifest {
  public enum Kernel: Int32, Sendable {
    case forward = 0
    case backwardQuery = 1
    case backwardKeyValue = 2
    case mlaCompressed = 3
  }

  public enum Key: String, CaseIterable, Sendable {
    case qData
    case kData
    case vData
    case output
    case gradOutput
    case logsumexp
    case gradQuery
    case dValues
    case gradKey
    case gradValue
    case qScale
    case qZeroPoint
    case kScale
    case kZeroPoint
    case vScale
    case vZeroPoint
    case dims
    case steClipRange
    case qBlockScales
    case qBlockZeroPoints
    case kBlockScales
    case kBlockZeroPoints
    case vBlockScales
    case vBlockZeroPoints
    case qPrecomputedSums
    case kPrecomputedSums
    case vPrecomputedSums
    case qStrides
    case kStrides
    case vStrides
    case oStrides
    case maskBuffer
    case maskMetadata
    case numHeads
    case numKeyValueHeads
    case headDimension
    case sequenceLength
    case scratch0
    case scratch1
  }

  /// Canonical slot numbers for each key.
  private static let slotTable: [Key: Int] = {
    var slots: [Key: Int] = [:]
    var next = 0
    func assign(_ key: Key, _ index: inout Int, _ slots: inout [Key: Int]) {
      slots[key] = index
      index += 1
    }

    func assignSame(_ keys: [Key], _ value: Int, _ slots: inout [Key: Int]) {
      for key in keys {
        slots[key] = value
      }
    }

    assign(.qData, &next, &slots) // 0
    assign(.kData, &next, &slots) // 1
    assign(.vData, &next, &slots) // 2

    let primaryOutputIndex = next
    assignSame([.output, .gradOutput], primaryOutputIndex, &slots)
    next += 1 // 3

    assign(.logsumexp, &next, &slots) // 4
    assign(.gradQuery, &next, &slots) // 5
    assign(.dValues, &next, &slots) // 6
    assign(.gradKey, &next, &slots) // 7
    assign(.gradValue, &next, &slots) // 8

    assign(.qScale, &next, &slots) // 9
    assign(.qZeroPoint, &next, &slots) // 10
    assign(.kScale, &next, &slots) // 11
    assign(.kZeroPoint, &next, &slots) // 12
    assign(.vScale, &next, &slots) // 13
    assign(.vZeroPoint, &next, &slots) // 14

    assign(.dims, &next, &slots) // 15
    assign(.steClipRange, &next, &slots) // 16

    assign(.qBlockScales, &next, &slots) // 17
    assign(.qBlockZeroPoints, &next, &slots) // 18
    assign(.kBlockScales, &next, &slots) // 19
    assign(.kBlockZeroPoints, &next, &slots) // 20
    assign(.vBlockScales, &next, &slots) // 21
    assign(.vBlockZeroPoints, &next, &slots) // 22

    assign(.qPrecomputedSums, &next, &slots) // 23
    assign(.kPrecomputedSums, &next, &slots) // 24
    assign(.vPrecomputedSums, &next, &slots) // 25

    assign(.qStrides, &next, &slots) // 26
    assign(.kStrides, &next, &slots) // 27
    assign(.vStrides, &next, &slots) // 28
    assign(.oStrides, &next, &slots) // 29

    assign(.maskBuffer, &next, &slots) // 30

    // Metal only supports buffer indices 0-30, so we've reached the limit.
    // For scalar metadata parameters, we'll remove them from the buffer layout
    // and pass them as function constants or pack into existing buffers.
    // scratch0 and scratch1 are removed from backward passes (unused).

    // Mark these as unavailable (-1) for kernels that don't use them
    slots[.numHeads] = -1
    slots[.numKeyValueHeads] = -1
    slots[.headDimension] = -1
    slots[.sequenceLength] = -1
    slots[.scratch0] = -1
    slots[.scratch1] = -1

    return slots
  }()

  /// Lightweight view that exposes typed subscripts.
  public struct Layout: Sendable {
    public let kernel: Kernel
    private let indices: [Key: Int]

    fileprivate init(kernel: Kernel, keys: [Key]) {
      self.kernel = kernel
      indices = Dictionary(uniqueKeysWithValues: keys.compactMap { key in
        guard let value = slotTable[key] else {
          return nil
        }
        return (key, value)
      })
    }

    public func index(_ key: Key) -> Int {
      indices[key] ?? -1
    }

    public func dictionary() -> [String: Int] {
      indices.reduce(into: [:]) { dict, element in
        dict[element.key.rawValue] = element.value
      }
    }
  }

  private static let forwardLayout = Layout(
    kernel: .forward,
    keys: [
      .qData, .kData, .vData, .output, .logsumexp,
      .qScale, .qZeroPoint, .kScale, .kZeroPoint, .vScale, .vZeroPoint,
      .qBlockScales, .qBlockZeroPoints, .kBlockScales, .kBlockZeroPoints,
      .vBlockScales, .vBlockZeroPoints,
      .qPrecomputedSums, .kPrecomputedSums, .vPrecomputedSums,
      .qStrides, .kStrides, .vStrides, .oStrides,
      .maskBuffer,
      .numHeads, .numKeyValueHeads, .headDimension, .sequenceLength,
      .scratch0, .scratch1,
    ]
  )

  private static let backwardQueryLayout = Layout(
    kernel: .backwardQuery,
    keys: [
      .qData, .kData, .vData, .gradOutput, .logsumexp,
      .gradQuery, .dValues,
      .qScale, .qZeroPoint, .kScale, .kZeroPoint, .vScale, .vZeroPoint,
      .dims, .steClipRange,
      .qBlockScales, .qBlockZeroPoints, .kBlockScales, .kBlockZeroPoints,
      .vBlockScales, .vBlockZeroPoints,
      .qStrides, .kStrides, .vStrides, .oStrides,
      // Removed: numHeads, numKeyValueHeads, headDimension, sequenceLength, scratch0, scratch1
      // These exceed Metal's buffer limit and are unused or can be derived from dims
    ]
  )

  private static let backwardKeyValueLayout = Layout(
    kernel: .backwardKeyValue,
    keys: [
      .qData, .kData, .vData, .gradOutput, .logsumexp,
      .dValues, .gradKey, .gradValue,
      .qScale, .qZeroPoint, .kScale, .kZeroPoint, .vScale, .vZeroPoint,
      .dims, .steClipRange,
      .qBlockScales, .qBlockZeroPoints, .kBlockScales, .kBlockZeroPoints,
      .vBlockScales, .vBlockZeroPoints,
      .qStrides, .kStrides, .vStrides, .oStrides,
      // Removed: numHeads, numKeyValueHeads, headDimension, sequenceLength, scratch0, scratch1
      // These exceed Metal's buffer limit and are unused or can be derived from dims
    ]
  )

  private static let mlaCompressedLayout = Layout(
    kernel: .mlaCompressed,
    keys: [
      // MLA has different operands: Q, KV_latent, W_decompress_k, W_decompress_v, O
      // For compatibility, map to standard buffer slots
      .qData, .output,
      .numHeads, .headDimension, .sequenceLength,
      .scratch0, .scratch1,
    ]
  )

  public static func layout(for kernel: Kernel) -> Layout {
    switch kernel {
    case .forward:
      forwardLayout
    case .backwardQuery:
      backwardQueryLayout
    case .backwardKeyValue:
      backwardKeyValueLayout
    case .mlaCompressed:
      mlaCompressedLayout
    }
  }
}

public extension QuantizedKernelLayoutManifest.Layout {
  var qData: Int { index(.qData) }
  var kData: Int { index(.kData) }
  var vData: Int { index(.vData) }
  var output: Int { index(.output) }
  var gradOutput: Int { index(.gradOutput) }
  var logsumexp: Int { index(.logsumexp) }
  var gradQuery: Int { index(.gradQuery) }
  var dValues: Int { index(.dValues) }
  var gradKey: Int { index(.gradKey) }
  var gradValue: Int { index(.gradValue) }
  var qScale: Int { index(.qScale) }
  var qZeroPoint: Int { index(.qZeroPoint) }
  var kScale: Int { index(.kScale) }
  var kZeroPoint: Int { index(.kZeroPoint) }
  var vScale: Int { index(.vScale) }
  var vZeroPoint: Int { index(.vZeroPoint) }
  var dims: Int { index(.dims) }
  var steClipRange: Int { index(.steClipRange) }
  var qBlockScales: Int { index(.qBlockScales) }
  var qBlockZeroPoints: Int { index(.qBlockZeroPoints) }
  var kBlockScales: Int { index(.kBlockScales) }
  var kBlockZeroPoints: Int { index(.kBlockZeroPoints) }
  var vBlockScales: Int { index(.vBlockScales) }
  var vBlockZeroPoints: Int { index(.vBlockZeroPoints) }
  var qPrecomputedSums: Int { index(.qPrecomputedSums) }
  var kPrecomputedSums: Int { index(.kPrecomputedSums) }
  var vPrecomputedSums: Int { index(.vPrecomputedSums) }
  var qStrides: Int { index(.qStrides) }
  var kStrides: Int { index(.kStrides) }
  var vStrides: Int { index(.vStrides) }
  var oStrides: Int { index(.oStrides) }
  var maskBuffer: Int { index(.maskBuffer) }
  var maskMetadata: Int { index(.maskMetadata) }
  var numHeads: Int { index(.numHeads) }
  var numKeyValueHeads: Int { index(.numKeyValueHeads) }
  var headDimension: Int { index(.headDimension) }
  var sequenceLength: Int { index(.sequenceLength) }
  var scratch0: Int { index(.scratch0) }
  var scratch1: Int { index(.scratch1) }
}

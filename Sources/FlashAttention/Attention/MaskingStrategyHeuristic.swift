import Foundation
@preconcurrency import Metal

/// Masking strategy for applying sparsity patterns (causal, sliding window,
/// block-sparse) inside the softmax.
///
/// `bitmask` precomputes the mask as a small integer bitmask and applies it
/// with bitwise logic; `elementWise` checks each element's (row, col) against
/// the sparsity rule directly. Which is faster depends on the problem shape
/// and the GPU — bitmask has lower arithmetic cost but element-wise avoids
/// the bitmask setup, and the crossover moves sharply with sequence length.
public enum MaskingStrategy: String, Codable, Sendable {
  case elementWise
  case bitmask
}

/// Shared, cached, data-driven selector for `MaskingStrategy`.
///
/// Both the kernel codegen (`AttentionKernel.shouldUseBitmaskOptimization`)
/// and downstream callers resolve through `MaskingStrategyHeuristic.shared` so
/// they always agree.
///
/// Resolution order for `recommend`:
/// 1. In-memory cache (populated by `recordMeasurement`, e.g. from a warm
///    benchmark on this device).
/// 2. A fitted default rule (derived from Apple-Silicon benchmarking).
///
/// The default rule is a cheap cold-start; call `recordMeasurement` to seed the
/// cache with measured truths for the shapes you care about. The cache is
/// keyed by a coarse sequence-length bucket so nearby sizes share a result.
public final class MaskingStrategyHeuristic: @unchecked Sendable {
  public static let shared = MaskingStrategyHeuristic()

  private var cache: [ShapeKey: MaskingStrategy] = [:]
  private let lock = NSLock()

  private struct ShapeKey: Hashable {
    let sequenceBucket: Int
    let headDimension: Int
  }

  /// Bucket a sequence length to a nearby calibrated anchor so that close
  /// sizes share one decision (and one cache entry).
  public static func sequenceBucket(_ sequenceLength: Int) -> Int {
    let anchors = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
    var best = anchors[0]
    var bestDelta = abs(sequenceLength - best)
    for anchor in anchors.dropFirst() {
      let delta = abs(sequenceLength - anchor)
      if delta < bestDelta {
        best = anchor
        bestDelta = delta
      }
    }
    return best
  }

  /// Recommend a masking strategy for the given shape.
  public func recommend(
    sequenceLength: Int,
    headDimension: Int
  )
    -> MaskingStrategy
  {
    let key = ShapeKey(
      sequenceBucket: Self.sequenceBucket(sequenceLength),
      headDimension: headDimension
    )
    lock.lock()
    if let cached = cache[key] {
      lock.unlock()
      return cached
    }
    let choice = defaultRule(sequenceLength: sequenceLength, headDimension: headDimension)
    cache[key] = choice
    lock.unlock()
    return choice
  }

  /// Record a measured strategy (e.g. the winner of a warm benchmark) so that
  /// later `recommend` calls for this shape return it from the cache instead of
  /// the default rule. Thread-safe.
  public func recordMeasurement(
    sequenceLength: Int,
    headDimension: Int,
    strategy: MaskingStrategy
  ) {
    let key = ShapeKey(
      sequenceBucket: Self.sequenceBucket(sequenceLength),
      headDimension: headDimension
    )
    lock.lock()
    cache[key] = strategy
    lock.unlock()
  }

  /// Clear the cache (e.g. between benchmark runs to force re-evaluation).
  public func clearCache() {
    lock.lock()
    cache.removeAll()
    lock.unlock()
  }

  /// Cold-start default rule, fitted to benchmark data on Apple Silicon.
  ///
  /// Key finding: the sequence length is the dominant signal (the previous
  /// head-only heuristic ignored it). Head 192 consistently favours bitmask;
  /// very large sequences (≥4096) favour element-wise except at head 192;
  /// seq≈2048 is a bitmask sweet spot; head 128 only favours bitmask for small
  /// sequences.
  private func defaultRule(
    sequenceLength: Int,
    headDimension: Int
  )
    -> MaskingStrategy
  {
    if headDimension == 192 {
      return .bitmask
    }
    if sequenceLength >= 4096 {
      return .elementWise
    }
    if sequenceLength >= 1792, sequenceLength <= 2560 {
      return .bitmask
    }
    if headDimension == 128 {
      return sequenceLength <= 256 ? .bitmask : .elementWise
    }
    if sequenceLength <= 256 {
      return .bitmask
    }
    if headDimension == 256, sequenceLength >= 1024 {
      return .elementWise
    }
    return .bitmask
  }

  // MARK: - Calibration payload

  /// Hydrate the in-memory cache from a previously-produced (or loaded from
  /// disk) calibration. Entries override the default rule for their shapes.
  public func apply(_ calibration: MaskingCalibration) {
    lock.lock()
    for entry in calibration.entries {
      cache[ShapeKey(sequenceBucket: entry.sequenceBucket, headDimension: entry.headDimension)] =
        entry.strategy
    }
    lock.unlock()
  }
}

/// A serializable record of calibrated masking-strategy decisions for a set of
/// shapes on one device.
///
/// Produced by `MaskingStrategyHeuristic.calibrate(...)`; consumable by
/// `MaskingStrategyHeuristic.apply(_:)` and by a separate persistence helper
/// (`MaskingCalibrationStore`). Callers that don't want to persist can simply
/// run calibration and let the result be discarded — the in-memory cache is
/// already populated as a side effect.
public struct MaskingCalibration: Codable {
  public struct Entry: Codable {
    public let sequenceBucket: Int
    public let headDimension: Int
    public let strategy: MaskingStrategy
    /// Measured kernel time (ms) for each strategy; carried as evidence.
    public let bitmaskMs: Double
    public let elementWiseMs: Double

    public init(
      sequenceBucket: Int,
      headDimension: Int,
      strategy: MaskingStrategy,
      bitmaskMs: Double,
      elementWiseMs: Double
    ) {
      self.sequenceBucket = sequenceBucket
      self.headDimension = headDimension
      self.strategy = strategy
      self.bitmaskMs = bitmaskMs
      self.elementWiseMs = elementWiseMs
    }
  }

  public let deviceName: String
  public let entries: [Entry]

  public init(deviceName: String, entries: [Entry]) {
    self.deviceName = deviceName
    self.entries = entries
  }
}

public extension MaskingStrategyHeuristic {
  /// Benchmark bitmask vs element-wise masking for each shape and return a
  /// `MaskingCalibration` payload. As a side effect, each measured winner is
  /// recorded into the in-memory cache, so callers who don't care about
  /// persisting the payload still benefit immediately.
  ///
  /// Pass the resulting `MaskingCalibration` to `MaskingCalibrationStore.save`
  /// if you want to reuse it across runs; otherwise just discard it.
  ///
  /// - Parameters:
  ///   - shapes: `(sequenceLength, headDimension)` pairs to calibrate.
  ///   - device: Device to benchmark on (defaults to `MTLContext.global`).
  ///   - warmupIterations: Warmup dispatches before timing (GPU clock ramp).
  ///   - trialIterations: Timed dispatches; the fastest is kept.
  func calibrate(
    shapes: [(sequenceLength: Int, headDimension: Int)],
    device: MTLDevice? = nil,
    warmupIterations: Int = 5,
    trialIterations: Int = 10
  )
    -> MaskingCalibration
  {
    let dev = device ?? MTLContext.global.device
    let queue = MTLContext.global.commandQueue
    var entries: [MaskingCalibration.Entry] = []

    for shape in shapes {
      let bitmaskMs = timeCausalForward(
        strategy: .bitmask, sequenceLength: shape.sequenceLength,
        headDimension: shape.headDimension, device: dev, queue: queue,
        warmupIterations: warmupIterations, trialIterations: trialIterations
      )
      let elementWiseMs = timeCausalForward(
        strategy: .elementWise, sequenceLength: shape.sequenceLength,
        headDimension: shape.headDimension, device: dev, queue: queue,
        warmupIterations: warmupIterations, trialIterations: trialIterations
      )

      guard let bm = bitmaskMs, let ew = elementWiseMs else {
        print(
          "calibrate: skipped seq=\(shape.sequenceLength) head=\(shape.headDimension) (pipeline failure)"
        )
        continue
      }
      let winner: MaskingStrategy = bm <= ew ? .bitmask : .elementWise
      entries.append(
        MaskingCalibration.Entry(
          sequenceBucket: Self.sequenceBucket(shape.sequenceLength),
          headDimension: shape.headDimension,
          strategy: winner,
          bitmaskMs: bm,
          elementWiseMs: ew
        )
      )
      recordMeasurement(
        sequenceLength: shape.sequenceLength,
        headDimension: shape.headDimension,
        strategy: winner
      )
    }

    return MaskingCalibration(deviceName: dev.name, entries: entries)
  }

  /// Explicitly warm up the heuristic for a set of shapes, amortizing the
  /// calibration across runs.
  ///
  /// - If `persistTo` points at an existing calibration for this device, it is
  ///   loaded and applied (cheap — no benchmarking).
  /// - Otherwise the shapes are benchmarked; if `persistTo` is non-nil the
  ///   result is saved there for next time.
  /// - Either way the in-memory cache is populated, so `recommend(...)` is
  ///   answered from measurement rather than the cold-start rule.
  ///
  /// Pass `persistTo: nil` to benchmark for the current process only (the
  /// "wasteful" path — payload still returned, just not saved).
  func warmUp(
    shapes: [(sequenceLength: Int, headDimension: Int)],
    persistTo url: URL? = nil,
    device: MTLDevice? = nil,
    warmupIterations: Int = 5,
    trialIterations: Int = 10
  )
    -> MaskingCalibration
  {
    let dev = device ?? MTLContext.global.device

    // Reuse a persisted calibration that matches this device, if present.
    if
      let url,
      let loaded = try? MaskingCalibrationStore.load(from: url),
      loaded.deviceName == dev.name,
      !loaded.entries.isEmpty
    {
      apply(loaded)
      return loaded
    }

    // Otherwise benchmark now and (optionally) persist for next time.
    let calibration = calibrate(
      shapes: shapes,
      device: dev,
      warmupIterations: warmupIterations,
      trialIterations: trialIterations
    )
    if let url {
      try? MaskingCalibrationStore.save(calibration, to: url)
    }
    return calibration
  }

  /// Time a causal forward kernel with the given masking strategy forced.
  /// Returns the fastest trial latency in milliseconds, or nil on pipeline
  /// failure.
  private func timeCausalForward(
    strategy: MaskingStrategy,
    sequenceLength: Int,
    headDimension: Int,
    device: MTLDevice,
    queue: MTLCommandQueue,
    warmupIterations: Int,
    trialIterations: Int
  )
    -> Double?
  {
    var desc = AttentionDescriptor()
    desc.lowPrecisionInputs = false
    desc.matrixDimensions = (
      row: UInt32(sequenceLength),
      column: UInt32(sequenceLength),
      head: UInt16(headDimension)
    )
    desc.transposeState = (Q: false, K: false, V: false, O: false)
    desc.sparsityPattern = .causal

    var kernelDescriptor = desc.kernelDescriptor(type: .forward)
    kernelDescriptor.maskingStrategyOverride = strategy
    let kernel = AttentionKernel(descriptor: kernelDescriptor)

    let functionConstants = MTLFunctionConstantValues()
    desc.setFunctionConstants(functionConstants)

    guard
      let library = try? device.makeLibrary(source: kernel.createSource(), options: .mfaDefault),
      let function = try? library.makeFunction(name: "attention", constantValues: functionConstants)
    else {
      return nil
    }
    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
    guard
      let pipeline = try? device.makeComputePipelineState(
        descriptor: pipelineDesc, options: [], reflection: nil
      )
    else {
      return nil
    }

    // FP32 single-head operands. Small nonzero values (avoid denormals/NaN).
    let n = sequenceLength * headDimension
    let fill = Array(repeating: Float(0.01), count: n)
    let zero = [Float](repeating: 0, count: n)
    let lzeros = [Float](repeating: 0, count: sequenceLength)
    guard
      let q = device.makeBuffer(bytes: fill, length: n * 4, options: .storageModeShared),
      let k = device.makeBuffer(bytes: fill, length: n * 4, options: .storageModeShared),
      let v = device.makeBuffer(bytes: fill, length: n * 4, options: .storageModeShared),
      let o = device.makeBuffer(bytes: zero, length: n * 4, options: .storageModeShared),
      let l = device.makeBuffer(
        bytes: lzeros,
        length: sequenceLength * 4,
        options: .storageModeShared
      )
    else {
      return nil
    }

    let parallelization = Int(kernel.blockDimensions.parallelization)
    let blocks = max(1, (sequenceLength + parallelization - 1) / parallelization)
    let gridSize = MTLSize(width: blocks, height: 1, depth: 1)
    let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)
    let tgMem = Int(kernel.threadgroupMemoryAllocation)

    func runOnce() -> Double {
      guard
        let cb = queue.makeCommandBuffer(),
        let enc = cb.makeComputeCommandEncoder()
      else {
        return .infinity
      }
      enc.setComputePipelineState(pipeline)
      enc.setThreadgroupMemoryLength(tgMem, index: 0)
      enc.setBuffer(q, offset: 0, index: 0)
      enc.setBuffer(k, offset: 0, index: 1)
      enc.setBuffer(v, offset: 0, index: 2)
      enc.setBuffer(o, offset: 0, index: 3)
      enc.setBuffer(l, offset: 0, index: 4)
      enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
      enc.endEncoding()
      cb.commit()
      cb.waitUntilCompleted()
      return cb.gpuEndTime - cb.gpuStartTime
    }

    for _ in 0..<warmupIterations {
      _ = runOnce()
    }
    var best = Double.infinity
    for _ in 0..<trialIterations {
      best = min(best, runOnce())
    }
    return best.isFinite ? best * 1000.0 : nil
  }
}

/// Separate persistence helper for `MaskingCalibration`.
///
/// Intentionally decoupled from the heuristic: callers that want to amortize a
/// calibration across runs write it to disk here and reload it later via
/// `MaskingStrategyHeuristic.apply(_:)`. Callers that don't care about
/// persistence never touch this and just use the in-memory cache.
public enum MaskingCalibrationStore {
  /// A conventional on-disk location for a device's calibration
  /// (`~/.cache/<bundle>/masking-calibration/<deviceName>.json`).
  public static func defaultURL(
    deviceName: String,
    bundleIdentifier: String = "FlashAttention"
  )
    -> URL
  {
    let sanitized = deviceName.replacingOccurrences(of: " ", with: "_")
    let home = FileManager.default.homeDirectoryForCurrentUser
    return home
      .appendingPathComponent(".cache", isDirectory: true)
      .appendingPathComponent(bundleIdentifier, isDirectory: true)
      .appendingPathComponent("masking-calibration", isDirectory: true)
      .appendingPathComponent("\(sanitized).json")
  }

  /// Encode a calibration to JSON at `url`.
  public static func save(_ calibration: MaskingCalibration, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(calibration)
    let dir = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try data.write(to: url, options: .atomic)
  }

  /// Decode a calibration previously written by `save(_:to:)`.
  public static func load(from url: URL) throws -> MaskingCalibration {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(MaskingCalibration.self, from: data)
  }
}

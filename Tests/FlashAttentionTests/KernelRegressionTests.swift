import FlashAttention
import Metal
import XCTest

/// Regression coverage for kernel behaviors that broke (or nearly broke) in
/// the field, validated against a CPU reference:
///
/// - transposed causal masking parity for BOTH masking strategies (the
///   bitmask generator once kept the inverted triangle, corrupting dK/dV on
///   heuristic-selected shapes only)
/// - pipeline cache keys distinguishing shapes that share generated source
///   (function constants are baked into the PSO; keying by source hash alone
///   reused wrong-shape pipelines)
/// - encodeForward into an external command buffer matching forward()
/// - strided Q/K/V views matching contiguous inputs
/// - bf16 inputs with BF16 registers matching an fp32 reference
///
/// Tests on large shapes are gated behind MFA_SLOW_TESTS=1 so the default
/// `swift test` stays within the memory/time budget of the 8 GB CI runners;
/// run them locally on a big machine:
///
///     MFA_SLOW_TESTS=1 swift test --filter KernelRegressionTests
final class KernelRegressionTests: XCTestCase {
  override func tearDown() {
    // Tests may pin a masking strategy for a shape; never leak that into
    // other tests (the heuristic cache is process-global).
    MaskingStrategyHeuristic.shared.clearCache()
    super.tearDown()
  }

  private func requireSlowTests() throws {
    guard ProcessInfo.processInfo.environment["MFA_SLOW_TESTS"] == "1" else {
      throw XCTSkip(
        "Slow test: exceeds the CI runner budget. Set MFA_SLOW_TESTS=1 to run locally."
      )
    }
  }

  // MARK: - Deterministic data

  private func deterministicData(count: Int, seed: UInt64, scale: Float = 0.25) -> [Float] {
    var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    var out = [Float](repeating: 0, count: count)
    for i in 0..<count {
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      let unit = Float(state >> 40) / Float(1 << 24)
      out[i] = (unit * 2 - 1) * scale
    }
    return out
  }

  private func bf16Bytes(_ data: [Float]) -> [UInt16] {
    data.map { value in
      var bits = value.bitPattern
      let lsb = (bits >> 16) & 1
      bits = bits &+ 0x7FFF &+ lsb
      return UInt16(truncatingIfNeeded: bits >> 16)
    }
  }

  // MARK: - CPU reference (BHSD, fp32)

  private struct Reference {
    var output: [Float]
    var dQuery: [Float]
    var dKey: [Float]
    var dValue: [Float]
  }

  /// Straightforward O(S^2 D) attention forward + backward per (batch, head).
  /// dOutput of all-ones is used for the backward comparison.
  private func referenceAttention(
    q: [Float], k: [Float], v: [Float],
    batch: Int, heads: Int, seq: Int, dim: Int,
    causal: Bool
  ) -> Reference {
    let scale = 1.0 / Float(dim).squareRoot()
    var output = [Float](repeating: 0, count: q.count)
    var dQ = [Float](repeating: 0, count: q.count)
    var dK = [Float](repeating: 0, count: q.count)
    var dV = [Float](repeating: 0, count: q.count)

    for b in 0..<batch {
      for h in 0..<heads {
        let base = (b * heads + h) * seq * dim
        var probs = [Float](repeating: 0, count: seq * seq)

        // S = scale * Q K^T (masked), P = softmax(S)
        for row in 0..<seq {
          var maxScore = -Float.greatestFiniteMagnitude
          var scores = [Float](repeating: -.greatestFiniteMagnitude, count: seq)
          let colLimit = causal ? row + 1 : seq
          for col in 0..<colLimit {
            var dot: Float = 0
            for d in 0..<dim {
              dot += q[base + row * dim + d] * k[base + col * dim + d]
            }
            let score = dot * scale
            scores[col] = score
            maxScore = max(maxScore, score)
          }
          var denom: Float = 0
          for col in 0..<colLimit {
            let e = exp(scores[col] - maxScore)
            probs[row * seq + col] = e
            denom += e
          }
          for col in 0..<colLimit {
            probs[row * seq + col] /= denom
          }
          for d in 0..<dim {
            var acc: Float = 0
            for col in 0..<colLimit {
              acc += probs[row * seq + col] * v[base + col * dim + d]
            }
            output[base + row * dim + d] = acc
          }
        }

        // Backward with dO = 1: dV = P^T dO; dS = P .* (dP - rowsum(dP .* P));
        // dQ = scale * dS K; dK = scale * dS^T Q.
        for row in 0..<seq {
          let colLimit = causal ? row + 1 : seq
          var dPRow = [Float](repeating: 0, count: seq)
          var inner: Float = 0
          for col in 0..<colLimit {
            var dp: Float = 0
            for d in 0..<dim {
              dp += v[base + col * dim + d] // dO = 1
            }
            dPRow[col] = dp
            inner += dp * probs[row * seq + col]
          }
          for col in 0..<colLimit {
            let p = probs[row * seq + col]
            let dS = p * (dPRow[col] - inner)
            for d in 0..<dim {
              dQ[base + row * dim + d] += scale * dS * k[base + col * dim + d]
              dK[base + col * dim + d] += scale * dS * q[base + row * dim + d]
              dV[base + col * dim + d] += p // dO = 1
            }
          }
        }
      }
    }
    return Reference(output: output, dQuery: dQ, dKey: dK, dValue: dV)
  }

  // MARK: - GPU harness

  private func makeDescriptor(
    batch: Int, heads: Int, seq: Int, dim: Int,
    causal: Bool, lowPrecision: Bool = false
  ) -> MultiHeadAttentionDescriptor {
    var base = AttentionDescriptor()
    base.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(dim))
    base.lowPrecisionInputs = lowPrecision
    base.lowPrecisionIntermediates = lowPrecision
    if lowPrecision {
      base.inputMemoryPrecision = .BF16
    }
    base.transposeState = (Q: false, K: false, V: false, O: false)
    base.sparsityPattern = causal ? .causal : .none
    base.softmaxScale = 1.0 / Float(dim).squareRoot()

    let shape = MultiHeadShape(
      batchSize: UInt32(batch),
      numHeads: UInt32(heads),
      sequenceLength: UInt32(seq),
      headDimension: UInt16(dim)
    )
    return MultiHeadAttentionDescriptor(
      baseDescriptor: base,
      queryShape: shape, keyShape: shape, valueShape: shape,
      broadcastMode: .standard,
      dispatchStrategy: .perBatch
    )
  }

  private func floatBuffer(_ device: MTLDevice, _ data: [Float]) -> MTLBuffer {
    data.withUnsafeBytes { raw in
      device.makeBuffer(bytes: raw.baseAddress!, length: raw.count)!
    }
  }

  private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
  }

  private func runForward(
    _ attention: MultiHeadAttention,
    device: MTLDevice,
    q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
    batch: Int, heads: Int, seq: Int, dim: Int,
    descriptor: MultiHeadAttentionDescriptor
  ) throws -> (output: MTLBuffer, logsumexp: MTLBuffer) {
    let elements = batch * heads * seq * dim
    let output = device.makeBuffer(length: elements * 4)!
    let logsumexp = device.makeBuffer(length: batch * heads * seq * 4)!
    let commandBuffer = try XCTUnwrap(
      attention.forward(
        query: q, key: k, value: v, output: output,
        logsumexp: logsumexp, descriptor: descriptor
      )
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertNil(commandBuffer.error)
    return (output, logsumexp)
  }

  private func assertClose(
    _ observed: [Float], _ expected: [Float], tolerance: Float,
    _ label: String
  ) {
    XCTAssertEqual(observed.count, expected.count, label)
    var worst: Float = 0
    var worstIndex = 0
    for i in 0..<observed.count {
      let delta = abs(observed[i] - expected[i])
      if delta > worst {
        worst = delta
        worstIndex = i
      }
    }
    XCTAssertLessThanOrEqual(
      worst, tolerance,
      "\(label): max deviation \(worst) at index \(worstIndex) exceeds \(tolerance)"
    )
  }

  // MARK: - Causal backward parity, both masking strategies

  /// The transposed bitmask causal generator once kept q <= kv instead of
  /// q >= kv. Only shapes whose heuristic picked bitmask were corrupted, so
  /// pin each strategy explicitly and verify dK/dV against the reference.
  private func validateCausalBackward(strategy: MaskingStrategy) throws {
    let device = MTLContext.global.device
    let (batch, heads, seq, dim) = (1, 2, 96, 64)
    let elements = batch * heads * seq * dim

    MaskingStrategyHeuristic.shared.recordMeasurement(
      sequenceLength: seq, headDimension: dim, strategy: strategy
    )
    defer { MaskingStrategyHeuristic.shared.clearCache() }

    let qData = deterministicData(count: elements, seed: 11)
    let kData = deterministicData(count: elements, seed: 22)
    let vData = deterministicData(count: elements, seed: 33)
    let reference = referenceAttention(
      q: qData, k: kData, v: vData,
      batch: batch, heads: heads, seq: seq, dim: dim, causal: true
    )

    let attention = MultiHeadAttention(device: device)
    let descriptor = makeDescriptor(
      batch: batch, heads: heads, seq: seq, dim: dim, causal: true
    )
    let q = floatBuffer(device, qData)
    let k = floatBuffer(device, kData)
    let v = floatBuffer(device, vData)
    let (output, logsumexp) = try runForward(
      attention, device: device, q: q, k: k, v: v,
      batch: batch, heads: heads, seq: seq, dim: dim, descriptor: descriptor
    )
    assertClose(
      readFloats(output, count: elements), reference.output,
      tolerance: 2e-3, "forward output (\(strategy))"
    )

    let dOutput = floatBuffer(device, [Float](repeating: 1, count: elements))
    let dQuery = device.makeBuffer(length: elements * 4)!
    let dKey = device.makeBuffer(length: elements * 4)!
    let dValue = device.makeBuffer(length: elements * 4)!
    let dBuffer = device.makeBuffer(length: batch * heads * seq * 4)!
    memset(dQuery.contents(), 0, elements * 4)
    memset(dKey.contents(), 0, elements * 4)
    memset(dValue.contents(), 0, elements * 4)
    memset(dBuffer.contents(), 0, batch * heads * seq * 4)

    let commandBuffer = try XCTUnwrap(
      attention.backward(
        query: q, key: k, value: v, output: output, dOutput: dOutput,
        logsumexp: logsumexp, dQuery: dQuery, dKey: dKey, dValue: dValue,
        dBuffer: dBuffer, descriptor: descriptor
      )
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertNil(commandBuffer.error)

    assertClose(
      readFloats(dQuery, count: elements), reference.dQuery,
      tolerance: 5e-3, "dQ (\(strategy))"
    )
    assertClose(
      readFloats(dKey, count: elements), reference.dKey,
      tolerance: 5e-3, "dK (\(strategy))"
    )
    assertClose(
      readFloats(dValue, count: elements), reference.dValue,
      tolerance: 5e-3, "dV (\(strategy))"
    )
  }

  func testCausalBackwardParityBitmaskStrategy() throws {
    try validateCausalBackward(strategy: .bitmask)
  }

  func testCausalBackwardParityElementwiseStrategy() throws {
    try validateCausalBackward(strategy: .elementWise)
  }

  // MARK: - Pipeline cache must distinguish shapes sharing source

  /// Two sequence lengths that resolve to the same block configuration
  /// generate identical source; the dimensions live in function constants.
  /// A source-hash-only cache key returns the first shape's pipeline for the
  /// second shape and corrupts memory.
  func testPipelineCacheDistinguishesSequenceLengths() throws {
    let device = MTLContext.global.device
    let attention = MultiHeadAttention(device: device)
    let (batch, heads, dim) = (1, 2, 64)

    for seq in [64, 80] {
      let elements = batch * heads * seq * dim
      let qData = deterministicData(count: elements, seed: 44)
      let kData = deterministicData(count: elements, seed: 55)
      let vData = deterministicData(count: elements, seed: 66)
      let reference = referenceAttention(
        q: qData, k: kData, v: vData,
        batch: batch, heads: heads, seq: seq, dim: dim, causal: true
      )
      let descriptor = makeDescriptor(
        batch: batch, heads: heads, seq: seq, dim: dim, causal: true
      )
      let (output, _) = try runForward(
        attention, device: device,
        q: floatBuffer(device, qData),
        k: floatBuffer(device, kData),
        v: floatBuffer(device, vData),
        batch: batch, heads: heads, seq: seq, dim: dim, descriptor: descriptor
      )
      assertClose(
        readFloats(output, count: elements), reference.output,
        tolerance: 2e-3, "seq=\(seq) through shared instance"
      )
    }
  }

  // MARK: - encodeForward == forward

  func testEncodeForwardMatchesForward() throws {
    let device = MTLContext.global.device
    let attention = MultiHeadAttention(device: device)
    let (batch, heads, seq, dim) = (2, 3, 64, 32)
    let elements = batch * heads * seq * dim

    let qData = deterministicData(count: elements, seed: 77)
    let kData = deterministicData(count: elements, seed: 88)
    let vData = deterministicData(count: elements, seed: 99)
    let descriptor = makeDescriptor(
      batch: batch, heads: heads, seq: seq, dim: dim, causal: false
    )
    let q = floatBuffer(device, qData)
    let k = floatBuffer(device, kData)
    let v = floatBuffer(device, vData)

    let (viaForward, _) = try runForward(
      attention, device: device, q: q, k: k, v: v,
      batch: batch, heads: heads, seq: seq, dim: dim, descriptor: descriptor
    )

    let viaEncode = device.makeBuffer(length: elements * 4)!
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
    XCTAssertTrue(
      attention.encodeForward(
        commandBuffer: commandBuffer,
        query: q, key: k, value: v, output: viaEncode,
        descriptor: descriptor
      )
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertNil(commandBuffer.error)

    assertClose(
      readFloats(viaEncode, count: elements),
      readFloats(viaForward, count: elements),
      tolerance: 0, "encodeForward vs forward"
    )
  }

  // MARK: - Strided Q/K/V views

  /// Lay the logical BHSD tensor out as contiguous BSHD and present it via
  /// element strides. Must match the contiguous run exactly.
  func testStridedInputsMatchContiguous() throws {
    let device = MTLContext.global.device
    let attention = MultiHeadAttention(device: device)
    let (batch, heads, seq, dim) = (2, 3, 96, 64)
    let elements = batch * heads * seq * dim

    let qData = deterministicData(count: elements, seed: 101)
    let kData = deterministicData(count: elements, seed: 202)
    let vData = deterministicData(count: elements, seed: 303)
    let descriptor = makeDescriptor(
      batch: batch, heads: heads, seq: seq, dim: dim, causal: false
    )

    // Contiguous BHSD baseline.
    let (baseline, _) = try runForward(
      attention, device: device,
      q: floatBuffer(device, qData),
      k: floatBuffer(device, kData),
      v: floatBuffer(device, vData),
      batch: batch, heads: heads, seq: seq, dim: dim, descriptor: descriptor
    )

    // Same logical values stored BSHD: element (b, h, s, d) lives at
    // ((b*S + s)*H + h)*D + d, i.e. BHSD element strides [S*H*D, D, H*D, 1].
    func toBSHD(_ bhsd: [Float]) -> [Float] {
      var out = [Float](repeating: 0, count: bhsd.count)
      for b in 0..<batch {
        for h in 0..<heads {
          for s in 0..<seq {
            for d in 0..<dim {
              out[((b * seq + s) * heads + h) * dim + d] =
                bhsd[((b * heads + h) * seq + s) * dim + d]
            }
          }
        }
      }
      return out
    }
    let strides: [Int64] = [
      Int64(seq * heads * dim), Int64(dim), Int64(heads * dim), 1,
    ]

    let strided = device.makeBuffer(length: elements * 4)!
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
    XCTAssertTrue(
      attention.encodeForward(
        commandBuffer: commandBuffer,
        query: floatBuffer(device, toBSHD(qData)),
        key: floatBuffer(device, toBSHD(kData)),
        value: floatBuffer(device, toBSHD(vData)),
        output: strided,
        queryStrides: strides, keyStrides: strides, valueStrides: strides,
        descriptor: descriptor
      )
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertNil(commandBuffer.error)

    assertClose(
      readFloats(strided, count: elements),
      readFloats(baseline, count: elements),
      tolerance: 0, "strided vs contiguous"
    )
  }

  // MARK: - BF16 inputs

  /// BF16 memory with BF16 Q/K/V registers (S stays FP32) against the fp32
  /// reference on the bf16-rounded values.
  func testBF16InputsMatchReference() throws {
    let device = MTLContext.global.device
    guard device.supportsFamily(.apple9) else {
      throw XCTSkip("BF16 register path requires Apple9 GPU features")
    }
    let attention = MultiHeadAttention(device: device)
    let (batch, heads, seq, dim) = (1, 2, 96, 64)
    let elements = batch * heads * seq * dim

    let qData = deterministicData(count: elements, seed: 404)
    let kData = deterministicData(count: elements, seed: 505)
    let vData = deterministicData(count: elements, seed: 606)

    // Reference runs on the values the kernel actually sees: bf16-rounded.
    func roundTrip(_ data: [Float]) -> [Float] {
      bf16Bytes(data).map { Float(bitPattern: UInt32($0) << 16) }
    }
    let reference = referenceAttention(
      q: roundTrip(qData), k: roundTrip(kData), v: roundTrip(vData),
      batch: batch, heads: heads, seq: seq, dim: dim, causal: false
    )

    func bf16Buffer(_ data: [Float]) -> MTLBuffer {
      bf16Bytes(data).withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count)!
      }
    }
    let descriptor = makeDescriptor(
      batch: batch, heads: heads, seq: seq, dim: dim, causal: false,
      lowPrecision: true
    )
    // O is always stored FP32 regardless of the input precision.
    let (output, _) = try runForward(
      attention, device: device,
      q: bf16Buffer(qData), k: bf16Buffer(kData), v: bf16Buffer(vData),
      batch: batch, heads: heads, seq: seq, dim: dim, descriptor: descriptor
    )
    assertClose(
      readFloats(output, count: elements), reference.output,
      tolerance: 5e-3, "bf16 forward vs rounded fp32 reference"
    )
  }

  // MARK: - Slow tests (MFA_SLOW_TESTS=1; validated on M3 Max)

  /// DiT-scale shape: bf16 kernel against the fp32 kernel on identical
  /// (bf16-rounded) inputs. Large buffers and long kernels — local only.
  func testSlowLargeShapeBF16MatchesFP32() throws {
    try requireSlowTests()
    let device = MTLContext.global.device
    guard device.supportsFamily(.apple9) else {
      throw XCTSkip("BF16 register path requires Apple9 GPU features")
    }
    let attention = MultiHeadAttention(device: device)
    let (batch, heads, seq, dim) = (1, 30, 4128, 128)
    let elements = batch * heads * seq * dim

    let qData = deterministicData(count: elements, seed: 707, scale: 0.1)
    let kData = deterministicData(count: elements, seed: 808, scale: 0.1)
    let vData = deterministicData(count: elements, seed: 909, scale: 0.1)
    func roundTrip(_ data: [Float]) -> [Float] {
      bf16Bytes(data).map { Float(bitPattern: UInt32($0) << 16) }
    }

    // fp32 kernel on bf16-rounded values.
    let fp32Descriptor = makeDescriptor(
      batch: batch, heads: heads, seq: seq, dim: dim, causal: false
    )
    let (fp32Output, _) = try runForward(
      attention, device: device,
      q: floatBuffer(device, roundTrip(qData)),
      k: floatBuffer(device, roundTrip(kData)),
      v: floatBuffer(device, roundTrip(vData)),
      batch: batch, heads: heads, seq: seq, dim: dim,
      descriptor: fp32Descriptor
    )

    // bf16 kernel on the same values.
    func bf16Buffer(_ data: [Float]) -> MTLBuffer {
      bf16Bytes(data).withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count)!
      }
    }
    let bf16Descriptor = makeDescriptor(
      batch: batch, heads: heads, seq: seq, dim: dim, causal: false,
      lowPrecision: true
    )
    let (bf16Output, _) = try runForward(
      attention, device: device,
      q: bf16Buffer(qData), k: bf16Buffer(kData), v: bf16Buffer(vData),
      batch: batch, heads: heads, seq: seq, dim: dim,
      descriptor: bf16Descriptor
    )

    assertClose(
      readFloats(bf16Output, count: elements),
      readFloats(fp32Output, count: elements),
      tolerance: 5e-3, "large-shape bf16 vs fp32"
    )
  }

  /// Long-sequence causal backward against the CPU reference. The reference
  /// is O(S^2 D) on one core — minutes in debug builds, so local only.
  func testSlowLongSequenceCausalBackwardParity() throws {
    try requireSlowTests()
    let device = MTLContext.global.device
    let (batch, heads, seq, dim) = (1, 2, 1024, 64)
    let elements = batch * heads * seq * dim

    let qData = deterministicData(count: elements, seed: 111)
    let kData = deterministicData(count: elements, seed: 222)
    let vData = deterministicData(count: elements, seed: 333)
    let reference = referenceAttention(
      q: qData, k: kData, v: vData,
      batch: batch, heads: heads, seq: seq, dim: dim, causal: true
    )

    let attention = MultiHeadAttention(device: device)
    let descriptor = makeDescriptor(
      batch: batch, heads: heads, seq: seq, dim: dim, causal: true
    )
    let q = floatBuffer(device, qData)
    let k = floatBuffer(device, kData)
    let v = floatBuffer(device, vData)
    let (output, logsumexp) = try runForward(
      attention, device: device, q: q, k: k, v: v,
      batch: batch, heads: heads, seq: seq, dim: dim, descriptor: descriptor
    )

    let dOutput = floatBuffer(device, [Float](repeating: 1, count: elements))
    let dQuery = device.makeBuffer(length: elements * 4)!
    let dKey = device.makeBuffer(length: elements * 4)!
    let dValue = device.makeBuffer(length: elements * 4)!
    let dBuffer = device.makeBuffer(length: batch * heads * seq * 4)!
    memset(dQuery.contents(), 0, elements * 4)
    memset(dKey.contents(), 0, elements * 4)
    memset(dValue.contents(), 0, elements * 4)
    memset(dBuffer.contents(), 0, batch * heads * seq * 4)

    let commandBuffer = try XCTUnwrap(
      attention.backward(
        query: q, key: k, value: v, output: output, dOutput: dOutput,
        logsumexp: logsumexp, dQuery: dQuery, dKey: dKey, dValue: dValue,
        dBuffer: dBuffer, descriptor: descriptor
      )
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertNil(commandBuffer.error)

    assertClose(
      readFloats(dKey, count: elements), reference.dKey,
      tolerance: 2e-2, "long-sequence dK"
    )
    assertClose(
      readFloats(dValue, count: elements), reference.dValue,
      tolerance: 2e-2, "long-sequence dV"
    )
  }
}

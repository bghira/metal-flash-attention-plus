//
//  GEMMRuntimeQuantization.swift
//  FlashAttention
//
//  Swift interface for launching fused blockwise quantization kernels
//

import Foundation
import Metal

/// GPU tensor-wise quantization kernels.
///
/// Deliberately avoids the `bfloat` type — BF16 is handled via `ushort`
/// + bit shift — so the source compiles with ANY Metal language version,
/// even after torch.mps lowers the device default. This keeps the init
/// fast (<100 ms) and reliable.
private let tensorWiseQuantizationSource: String = """
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Compute tensor-wide absolute maximum via grid-wide atomic reduction.
// For non-negative floats, as_type<uint>(fabs(x)) is monotonically
// increasing, so atomic_fetch_max on the uint bits == float max.
kernel void compute_abs_max_fp16(
    device const half* input [[buffer(0)]],
    device atomic_uint* abs_max_bits [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint total [[threads_per_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint local = 0;
    for (uint i = tid; i < count; i += total) {
        float v = fabs(float(input[i]));
        local = max(local, as_type<uint>(v));
    }
    uint sm = simd_max(local);
    if (lane == 0) atomic_fetch_max_explicit(abs_max_bits, sm, memory_order_relaxed);
}

kernel void compute_abs_max_fp32(
    device const float* input [[buffer(0)]],
    device atomic_uint* abs_max_bits [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint total [[threads_per_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint local = 0;
    for (uint i = tid; i < count; i += total) {
        float v = fabs(input[i]);
        local = max(local, as_type<uint>(v));
    }
    uint sm = simd_max(local);
    if (lane == 0) atomic_fetch_max_explicit(abs_max_bits, sm, memory_order_relaxed);
}

// BF16 via ushort — avoids bfloat type dependency entirely.
// BF16 is the upper 16 bits of FP32: shift left 16, reinterpret as float.
kernel void compute_abs_max_bf16(
    device const ushort* input [[buffer(0)]],
    device atomic_uint* abs_max_bits [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint total [[threads_per_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint local = 0;
    for (uint i = tid; i < count; i += total) {
        float v = fabs(as_type<float>(uint(input[i]) << 16));
        local = max(local, as_type<uint>(v));
    }
    uint sm = simd_max(local);
    if (lane == 0) atomic_fetch_max_explicit(abs_max_bits, sm, memory_order_relaxed);
}

// ---- INT8 quantize (1 byte per element) ----

kernel void quantize_tw_fp16_to_int8(
    device const half* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device const uint* abs_max_bits [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float abs_max = as_type<float>(abs_max_bits[0]);
    float scale = max(abs_max / 127.0f, 1e-8f);
    int q = int(round(float(input[gid]) / scale));
    output[gid] = int8_t(clamp(q, -128, 127));
}

kernel void quantize_tw_fp32_to_int8(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device const uint* abs_max_bits [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float abs_max = as_type<float>(abs_max_bits[0]);
    float scale = max(abs_max / 127.0f, 1e-8f);
    int q = int(round(input[gid] / scale));
    output[gid] = int8_t(clamp(q, -128, 127));
}

kernel void quantize_tw_bf16_to_int8(
    device const ushort* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device const uint* abs_max_bits [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float abs_max = as_type<float>(abs_max_bits[0]);
    float scale = max(abs_max / 127.0f, 1e-8f);
    float v = as_type<float>(uint(input[gid]) << 16);
    int q = int(round(v / scale));
    output[gid] = int8_t(clamp(q, -128, 127));
}

// ---- INT4 quantize (packed 2 per byte) ----

kernel void quantize_tw_fp16_to_int4(
    device const half* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device const uint* abs_max_bits [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint base = gid * 2;
    if (base >= count) return;
    float abs_max = as_type<float>(abs_max_bits[0]);
    float scale = max(abs_max / 7.0f, 1e-8f);
    int q0 = int(round(float(input[base]) / scale));
    q0 = clamp(q0, -8, 7) + 8;
    int q1 = (base + 1 < count) ? int(round(float(input[base + 1]) / scale)) : 0;
    q1 = clamp(q1, -8, 7) + 8;
    output[gid] = uchar(q0 | (q1 << 4));
}

kernel void quantize_tw_fp32_to_int4(
    device const float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device const uint* abs_max_bits [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint base = gid * 2;
    if (base >= count) return;
    float abs_max = as_type<float>(abs_max_bits[0]);
    float scale = max(abs_max / 7.0f, 1e-8f);
    int q0 = int(round(input[base] / scale));
    q0 = clamp(q0, -8, 7) + 8;
    int q1 = (base + 1 < count) ? int(round(input[base + 1] / scale)) : 0;
    q1 = clamp(q1, -8, 7) + 8;
    output[gid] = uchar(q0 | (q1 << 4));
}

kernel void quantize_tw_bf16_to_int4(
    device const ushort* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device const uint* abs_max_bits [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint base = gid * 2;
    if (base >= count) return;
    float abs_max = as_type<float>(abs_max_bits[0]);
    float scale = max(abs_max / 7.0f, 1e-8f);
    float v0 = as_type<float>(uint(input[base]) << 16);
    int q0 = int(round(v0 / scale));
    q0 = clamp(q0, -8, 7) + 8;
    int q1 = 0;
    if (base + 1 < count) {
        float v1 = as_type<float>(uint(input[base + 1]) << 16);
        q1 = clamp(int(round(v1 / scale)), -8, 7) + 8;
    }
    output[gid] = uchar(q0 | (q1 << 4));
}
"""

/// Runtime quantization utilities for efficient GPU-based quantization
public class GEMMRuntimeQuantization {
  private let device: MTLDevice
  /// Always-available library with tensor-wise kernels (no bfloat dependency).
  private let tensorWiseLibrary: MTLLibrary
  /// Optional library with blockwise centered kernels (needs bfloat → may be
  /// nil when torch.mps has lowered the default language version).
  private let blockwiseLibrary: MTLLibrary?
  private var pipelineCache: [String: MTLComputePipelineState] = [:]

  public init(device: MTLDevice) throws {
    self.device = device

    // Tensor-wise library: compile the small self-contained source with
    // default options. No bfloat, no __asm — works with any language version.
    // Takes <100 ms.
    self.tensorWiseLibrary = try MetalLibraryCompiler.makeLibrary(
      device: device, source: tensorWiseQuantizationSource, options: nil)

    // Blockwise library: try makeDefaultLibrary (compiles bundled .metal
    // files). This may fail when torch.mps has lowered the device default
    // language version (disabling bfloat). That's OK — blockwise falls back
    // to CPU, and tensor-wise GPU still works.
    self.blockwiseLibrary = try? device.makeDefaultLibrary(bundle: Bundle.module)
  }

  /// Errors that can occur during quantization
  public enum QuantizationError: Error {
    case libraryNotFound
    case functionNotFound(String)
    case pipelineCreationFailed(String)
    case bufferCreationFailed
    case invalidParameters(String)
  }

  /// Get or create compute pipeline, looking up from the correct library.
  private func getPipeline(for kernelName: String) throws -> MTLComputePipelineState {
    if let cached = pipelineCache[kernelName] {
      return cached
    }

    // Tensor-wise kernels live in tensorWiseLibrary; everything else
    // (blockwise centered) in blockwiseLibrary.
    let isTensorWise = kernelName.hasPrefix("compute_abs_max_")
      || kernelName.hasPrefix("quantize_tw_")
    let lib = isTensorWise
      ? tensorWiseLibrary
      : (blockwiseLibrary ?? tensorWiseLibrary)

    guard let function = lib.makeFunction(name: kernelName) else {
      throw QuantizationError.functionNotFound(kernelName)
    }

    do {
      let pipeline = try device.makeComputePipelineState(function: function)
      pipelineCache[kernelName] = pipeline
      return pipeline
    } catch {
      throw QuantizationError.pipelineCreationFailed(kernelName)
    }
  }

  /// Perform fused blockwise centered quantization
  public func quantizeBlockwiseCentered(
    input: MTLBuffer,
    inputPrecision: GEMMOperandPrecision,
    output: MTLBuffer,
    blockScales: MTLBuffer,
    blockZeroPoints: MTLBuffer,
    precomputedSums: MTLBuffer?,
    K: Int,
    blockSizeK: Int,
    commandBuffer: MTLCommandBuffer
  ) throws {
    guard blockSizeK > 0, blockSizeK % 8 == 0 else {
      throw QuantizationError
        .invalidParameters("blockSizeK must be positive and multiple of 8")
    }
    guard K > 0 else {
      throw QuantizationError.invalidParameters("K must be positive")
    }

    let kernelName: String
    switch inputPrecision {
    case .FP32: kernelName = "quantize_blockwise_centered_fp32_to_int8"
    case .FP16: kernelName = "quantize_blockwise_centered_fp16_to_int8"
    case .BF16: kernelName = "quantize_blockwise_centered_bf16_to_int8"
    default:
      throw QuantizationError
        .invalidParameters("Unsupported input precision: \(inputPrecision)")
    }

    let pipeline = try getPipeline(for: kernelName)

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw QuantizationError.pipelineCreationFailed("Failed to create compute encoder")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBuffer(blockScales, offset: 0, index: 2)
    encoder.setBuffer(blockZeroPoints, offset: 0, index: 3)
    if let precomputedSums {
      encoder.setBuffer(precomputedSums, offset: 0, index: 4)
    } else {
      encoder.setBuffer(input, offset: 0, index: 4)
    }

    var K_uint = UInt32(K)
    var blockSizeK_uint = UInt32(blockSizeK)
    encoder.setBytes(&K_uint, length: MemoryLayout<UInt32>.size, index: 5)
    encoder.setBytes(&blockSizeK_uint, length: MemoryLayout<UInt32>.size, index: 6)

    let numBlocks = (K + blockSizeK - 1) / blockSizeK
    let threadsPerBlock = min(256, blockSizeK)
    let totalThreads = numBlocks * threadsPerBlock
    encoder.dispatchThreadgroups(
      MTLSize(width: (totalThreads + threadsPerBlock - 1) / threadsPerBlock, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: threadsPerBlock, height: 1, depth: 1))
    encoder.endEncoding()
  }

  /// Create buffers needed for blockwise quantization
  public func createBlockwiseBuffers(
    elementCount: Int,
    blockSizeK: Int,
    includePrecomputedSums: Bool = false
  ) throws -> (scales: MTLBuffer, zeroPoints: MTLBuffer, sums: MTLBuffer?) {
    let numBlocks = (elementCount + blockSizeK - 1) / blockSizeK
    guard let scalesBuffer = device.makeBuffer(
      length: numBlocks * MemoryLayout<Float>.size, options: .storageModeShared),
      let zeroPointsBuffer = device.makeBuffer(
        length: numBlocks * MemoryLayout<Int8>.size, options: .storageModeShared)
    else {
      throw QuantizationError.bufferCreationFailed
    }
    let sumsBuffer: MTLBuffer?
    if includePrecomputedSums {
      sumsBuffer = device.makeBuffer(
        length: numBlocks * MemoryLayout<Int32>.size, options: .storageModeShared)
      guard sumsBuffer != nil else { throw QuantizationError.bufferCreationFailed }
    } else {
      sumsBuffer = nil
    }
    return (scales: scalesBuffer, zeroPoints: zeroPointsBuffer, sums: sumsBuffer)
  }

  /// Extract quantization parameters from blockwise buffers
  public func extractBlockwiseParameters(
    scalesBuffer: MTLBuffer, zeroPointsBuffer: MTLBuffer, numBlocks: Int
  ) -> (scales: [Float], zeroPoints: [Int32]) {
    let scalesPtr = scalesBuffer.contents().bindMemory(to: Float.self, capacity: numBlocks)
    let zeroPointsPtr = zeroPointsBuffer.contents().bindMemory(to: Int8.self, capacity: numBlocks)
    return (
      scales: Array(UnsafeBufferPointer(start: scalesPtr, count: numBlocks)),
      zeroPoints: Array(UnsafeBufferPointer(start: zeroPointsPtr, count: numBlocks)).map { Int32($0) }
    )
  }

  /// Utility method to quantize using fused blockwise centered quantization
  public func quantizeBlockwiseCenteredTensor(
    inputBuffer: MTLBuffer,
    inputPrecision: GEMMOperandPrecision,
    elementCount: Int,
    blockSizeK: Int,
    commandBuffer: MTLCommandBuffer
  ) throws -> QuantizedTensor {
    guard let quantizedBuffer = device.makeBuffer(length: elementCount, options: .storageModeShared)
    else { throw QuantizationError.bufferCreationFailed }

    let (scalesBuffer, zeroPointsBuffer, _) = try createBlockwiseBuffers(
      elementCount: elementCount, blockSizeK: blockSizeK, includePrecomputedSums: false)

    try quantizeBlockwiseCentered(
      input: inputBuffer, inputPrecision: inputPrecision, output: quantizedBuffer,
      blockScales: scalesBuffer, blockZeroPoints: zeroPointsBuffer,
      precomputedSums: nil, K: elementCount, blockSizeK: blockSizeK,
      commandBuffer: commandBuffer)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let numBlocks = (elementCount + blockSizeK - 1) / blockSizeK
    let (scales, zeroPoints) = extractBlockwiseParameters(
      scalesBuffer: scalesBuffer, zeroPointsBuffer: zeroPointsBuffer, numBlocks: numBlocks)

    let quantParams = QuantizationParameters(
      scales: scales, zeroPoints: zeroPoints,
      precision: .INT8,
      mode: .blockwise(blockSizeK: blockSizeK, bothOperands: false),
      strategy: .symmetric)

    return QuantizedTensor(
      device: device, data: quantizedBuffer, parameters: quantParams,
      elementCount: elementCount, shape: [elementCount],
      blockScales: scalesBuffer, blockZeroPoints: zeroPointsBuffer,
      blockSizeK: blockSizeK, precomputedSums: nil)
  }

  /// GPU tensor-wise (per-tensor) symmetric quantization.
  ///
  /// Two-dispatch pipeline in one command buffer:
  ///  1. Grid-wide atomic reduction to compute abs-max.
  ///  2. Apply quantization using the computed scale.
  ///
  /// Commits + waits to read back the single scale value (needed for
  /// `setBytes` in the attention kernel). Adds ~1 ms of sync overhead but
  /// eliminates the multi-second CPU path.
  public func quantizeTensorWise(
    inputBuffer: MTLBuffer,
    inputPrecision: GEMMOperandPrecision,
    elementCount: Int,
    targetPrecision: GEMMOperandPrecision,
    into commandBuffer: MTLCommandBuffer? = nil
  ) throws -> (tensor: QuantizedTensor, commandBuffer: MTLCommandBuffer) {
    let cmdBuf = commandBuffer ?? device.makeCommandQueue()!.makeCommandBuffer()!

    guard let absMaxBuffer = device.makeBuffer(length: 4, options: .storageModeShared) else {
      throw QuantizationError.bufferCreationFailed
    }
    memset(absMaxBuffer.contents(), 0, 4)

    let isINT4 = targetPrecision == .INT4
    let outputSize = isINT4 ? (elementCount + 1) / 2 : elementCount
    guard let quantizedBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared)
    else { throw QuantizationError.bufferCreationFailed }

    let precName: String
    switch inputPrecision {
    case .FP16: precName = "fp16"
    case .FP32: precName = "fp32"
    case .BF16: precName = "bf16"
    default: throw QuantizationError.invalidParameters("Unsupported precision: \(inputPrecision)")
    }
    let targetName = isINT4 ? "int4" : "int8"

    // Dispatch 1: abs-max reduction
    let reductionPipeline = try getPipeline(for: "compute_abs_max_\(precName)")
    guard let enc1 = cmdBuf.makeComputeCommandEncoder() else {
      throw QuantizationError.pipelineCreationFailed("reduction encoder")
    }
    enc1.setComputePipelineState(reductionPipeline)
    enc1.setBuffer(inputBuffer, offset: 0, index: 0)
    enc1.setBuffer(absMaxBuffer, offset: 0, index: 1)
    var count = UInt32(elementCount)
    enc1.setBytes(&count, length: 4, index: 2)
    let tgSize = 256
    let groups = (elementCount + tgSize - 1) / tgSize
    enc1.dispatchThreadgroups(
      MTLSize(width: max(groups, 1), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    enc1.endEncoding()

    // Dispatch 2: quantize
    let quantPipeline = try getPipeline(for: "quantize_tw_\(precName)_to_\(targetName)")
    guard let enc2 = cmdBuf.makeComputeCommandEncoder() else {
      throw QuantizationError.pipelineCreationFailed("quantize encoder")
    }
    enc2.setComputePipelineState(quantPipeline)
    enc2.setBuffer(inputBuffer, offset: 0, index: 0)
    enc2.setBuffer(quantizedBuffer, offset: 0, index: 1)
    enc2.setBuffer(absMaxBuffer, offset: 0, index: 2)
    enc2.setBytes(&count, length: 4, index: 3)
    let workItems = isINT4 ? (elementCount + 1) / 2 : elementCount
    enc2.dispatchThreads(
      MTLSize(width: workItems, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    enc2.endEncoding()

    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let absMaxBits = absMaxBuffer.contents().load(as: UInt32.self)
    let absMax = Float(bitPattern: absMaxBits)
    let denom: Float = isINT4 ? 7.0 : 127.0
    let scale = max(absMax / denom, 1e-8)

    let parameters = QuantizationParameters(
      scale: scale, zeroPoint: 0,
      precision: targetPrecision,
      mode: .tensorWise,
      strategy: .symmetric)

    let tensor = QuantizedTensor(
      device: device, data: quantizedBuffer, parameters: parameters,
      elementCount: elementCount, shape: [elementCount])

    return (tensor, cmdBuf)
  }
}

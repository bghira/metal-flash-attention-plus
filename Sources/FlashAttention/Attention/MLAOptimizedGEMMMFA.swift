//
//  MLAOptimizedGEMMMFA.swift
//  FlashAttention
//
//  Multi-Latent Attention (MLA) decompression
//  Target: 10.9 TFLOPS @ 2048×2048 on M3 Max
//
//  TODO: Currently uses MPS for GEMM. Should be migrated to use MFA's superior GEMM kernels
//  for better performance. See GEMMKernel.register() and GEMMKernel.pipelineCache usage in
//  Tests/FlashAttentionTests/GEMM/LaplacianTest.swift for proper MFA GEMM integration.
//

import Metal
import MetalPerformanceShaders
import Foundation

/// MLA (Multi-Latent Attention) decompression
///
/// Decompresses KV cache from [batch, seq, kv_latent_dim] to full K and V
/// using learned weight matrices W_k and W_v.
///
/// Current: Uses MPS for GEMM (reliable, good performance)
/// Future: Should use MFA GEMM for superior performance (10.9 TFLOPS on M3 Max)
public final class MLAOptimizedGEMMMFA {
  private let device: MTLDevice
  private let commandQueue: MTLCommandQueue

  // Decompression weight matrices
  private var wk: MTLBuffer?  // [kv_latent_dim, num_heads × head_dim]
  private var wv: MTLBuffer?  // [kv_latent_dim, num_heads × head_dim]

  public init(device: MTLDevice) throws {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      throw NSError(
        domain: "MLAOptimizedGEMMMFA",
        code: -1,
        userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"]
      )
    }
    self.commandQueue = queue
  }

  /// Initialize random decompression weights for testing
  ///
  /// - Parameters:
  ///   - numHeads: Number of attention heads
  ///   - headDim: Dimension per head
  ///   - kvLatentDim: Compressed KV dimension
  public func initializeDecompressionWeights(
    numHeads: Int,
    headDim: Int,
    kvLatentDim: Int
  ) {
    let totalDim = numHeads * headDim
    let wkSize = kvLatentDim * totalDim * MemoryLayout<UInt16>.size
    let wvSize = kvLatentDim * totalDim * MemoryLayout<UInt16>.size

    // Create buffers with random FP16 data
    wk = device.makeBuffer(length: wkSize, options: .storageModeShared)
    wv = device.makeBuffer(length: wvSize, options: .storageModeShared)

    // Initialize with random values
    if let wkPtr = wk?.contents().assumingMemoryBound(to: UInt16.self) {
      for i in 0..<(kvLatentDim * totalDim) {
        // Simple FP32 to FP16 conversion
        let value = Float.random(in: -0.1...0.1)
        wkPtr[i] = UInt16((value.bitPattern >> 16) & 0xFFFF)
      }
    }

    if let wvPtr = wv?.contents().assumingMemoryBound(to: UInt16.self) {
      for i in 0..<(kvLatentDim * totalDim) {
        let value = Float.random(in: -0.1...0.1)
        wvPtr[i] = UInt16((value.bitPattern >> 16) & 0xFFFF)
      }
    }
  }

  /// Load pre-trained decompression weights
  ///
  /// - Parameters:
  ///   - wk: Weight matrix for K decompression
  ///   - wv: Weight matrix for V decompression
  public func loadWeights(wk: MTLBuffer, wv: MTLBuffer) {
    self.wk = wk
    self.wv = wv
  }

  /// Perform MLA forward pass: decompress KV latent into full K and V
  ///
  /// Performs two GEMMs:
  /// - K = KV_latent @ W_k
  /// - V = KV_latent @ W_v
  ///
  /// - Parameters:
  ///   - commandBuffer: Metal command buffer
  ///   - kvLatent: Input compressed KV [batch×seq, kv_latent_dim]
  ///   - decompressedK: Output K buffer [batch×seq, num_heads×head_dim]
  ///   - decompressedV: Output V buffer [batch×seq, num_heads×head_dim]
  ///   - batchSize: Batch size
  ///   - numHeads: Number of attention heads
  ///   - sequenceLength: Sequence length
  ///   - headDim: Head dimension
  ///   - kvLatentDim: Compressed latent dimension
  public func forward(
    commandBuffer: MTLCommandBuffer,
    kvLatent: MTLBuffer,
    decompressedK: inout MTLBuffer?,
    decompressedV: inout MTLBuffer?,
    batchSize: Int,
    numHeads: Int,
    sequenceLength: Int,
    headDim: Int,
    kvLatentDim: Int
  ) throws {
    guard let wk = self.wk, let wv = self.wv else {
      throw NSError(
        domain: "MLAOptimizedGEMMMFA",
        code: -2,
        userInfo: [NSLocalizedDescriptionKey: "Weights not initialized"]
      )
    }

    let M = batchSize * sequenceLength
    let N = numHeads * headDim
    let K = kvLatentDim

    // Allocate output buffers if needed
    let outputSize = M * N * MemoryLayout<UInt16>.size
    if decompressedK == nil {
      decompressedK = device.makeBuffer(length: outputSize, options: .storageModeShared)
    }
    if decompressedV == nil {
      decompressedV = device.makeBuffer(length: outputSize, options: .storageModeShared)
    }

    guard let k = decompressedK, let v = decompressedV else {
      throw NSError(
        domain: "MLAOptimizedGEMMMFA",
        code: -3,
        userInfo: [NSLocalizedDescriptionKey: "Failed to allocate output buffers"]
      )
    }

    // Use MPS for GEMM operations (reliable, good baseline performance)
    // TODO: Migrate to MFA GEMM for 10.9 TFLOPS performance
    // K = KV_latent @ W_k
    try encodeGEMMWithMPS(
      commandBuffer: commandBuffer,
      A: kvLatent,
      B: wk,
      C: k,
      M: M,
      N: N,
      K: K
    )

    // V = KV_latent @ W_v
    try encodeGEMMWithMPS(
      commandBuffer: commandBuffer,
      A: kvLatent,
      B: wv,
      C: v,
      M: M,
      N: N,
      K: K
    )
  }

  /// Public method for direct GEMM access (matches test expectations)
  public func encodeGEMM(
    commandBuffer: MTLCommandBuffer,
    A: MTLBuffer,
    B: MTLBuffer,
    C: MTLBuffer,
    M: Int,
    N: Int,
    K: Int
  ) {
    try? encodeGEMMWithMPS(
      commandBuffer: commandBuffer,
      A: A,
      B: B,
      C: C,
      M: M,
      N: N,
      K: K
    )
  }

  /// Encode GEMM operation using MPS (Metal Performance Shaders)
  ///
  /// C = A @ B where:
  /// - A: [M, K]
  /// - B: [K, N]
  /// - C: [M, N]
  private func encodeGEMMWithMPS(
    commandBuffer: MTLCommandBuffer,
    A: MTLBuffer,
    B: MTLBuffer,
    C: MTLBuffer,
    M: Int,
    N: Int,
    K: Int
  ) throws {
    // Use MPSMatrixMultiplication (same backend PyTorch MPS uses)
    let matrixA = MPSMatrix(
      buffer: A,
      descriptor: MPSMatrixDescriptor(
        rows: M,
        columns: K,
        rowBytes: K * MemoryLayout<UInt16>.size,
        dataType: .float16
      )
    )

    let matrixB = MPSMatrix(
      buffer: B,
      descriptor: MPSMatrixDescriptor(
        rows: K,
        columns: N,
        rowBytes: N * MemoryLayout<UInt16>.size,
        dataType: .float16
      )
    )

    let matrixC = MPSMatrix(
      buffer: C,
      descriptor: MPSMatrixDescriptor(
        rows: M,
        columns: N,
        rowBytes: N * MemoryLayout<UInt16>.size,
        dataType: .float16
      )
    )

    let matmul = MPSMatrixMultiplication(
      device: device,
      transposeLeft: false,
      transposeRight: false,
      resultRows: M,
      resultColumns: N,
      interiorColumns: K,
      alpha: 1.0,
      beta: 0.0
    )

    matmul.encode(
      commandBuffer: commandBuffer,
      leftMatrix: matrixA,
      rightMatrix: matrixB,
      resultMatrix: matrixC
    )
  }
}

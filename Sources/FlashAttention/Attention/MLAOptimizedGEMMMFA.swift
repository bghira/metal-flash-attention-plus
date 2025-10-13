//
//  MLAOptimizedGEMMMFA.swift
//  FlashAttention
//
//  MLA decompression using MFA's optimized GEMM code generation
//  Target: 8.5 TFLOPS FP32, 10 TFLOPS FP16 (vs MPS 7.5/7.0)
//

import Metal

/// MLA decompression using MFA's GEMMKernel code generation
public class MLAOptimizedGEMMMFA {
  private let device: MTLDevice

  // Decompression weights
  private var wDecompressK: MTLBuffer?
  private var wDecompressV: MTLBuffer?

  // Temporary buffers
  private var decompressedK: MTLBuffer?
  private var decompressedV: MTLBuffer?
  private var currentKVSize: Int = 0

  // Common MLA sizes to pre-register
  private let commonSizes: [(M: Int, N: Int, K: Int)] = [
    (M: 512, N: 512, K: 512),
    (M: 1024, N: 1024, K: 1024),
    (M: 2048, N: 2048, K: 2048),
    (M: 512, N: 512, K: 1024), // Asymmetric sizes
    (M: 1024, N: 512, K: 512),
  ]

  public init(device: MTLDevice? = nil) throws {
    self.device = device ?? MTLCreateSystemDefaultDevice()!

    // Pre-register common MLA sizes for immediate availability
    for size in commonSizes {
      registerGEMMKernel(M: size.M, N: size.N, K: size.K, precision: .FP16)
    }
  }

  /// Register and compile a GEMM kernel for given dimensions
  private func registerGEMMKernel(
    M: Int, N: Int, K: Int,
    precision: GEMMOperandPrecision
  ) {
    var desc = GEMMDescriptor()
    desc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
    desc.memoryPrecisions = (A: precision, B: precision, C: precision)
    desc.transposeState = (A: false, B: false)
    desc.loadPreviousC = false

    // Register (generates and compiles shader)
    GEMMKernel.register(descriptor: desc)
  }

  public func initializeDecompressionWeights(
    numHeads: Int,
    headDim: Int,
    kvLatentDim: Int
  ) {
    let totalDim = numHeads * headDim
    let bufferSize = kvLatentDim * totalDim * MemoryLayout<Float16>.size

    wDecompressK = device.makeBuffer(
      length: bufferSize, options: .storageModeShared
    )
    wDecompressV = device.makeBuffer(
      length: bufferSize, options: .storageModeShared
    )

    // Initialize with random weights (Xavier/Glorot)
    if
      let kPtr = wDecompressK?.contents().bindMemory(
        to: Float16.self, capacity: kvLatentDim * totalDim
      ),
      let vPtr = wDecompressV?.contents().bindMemory(
        to: Float16.self, capacity: kvLatentDim * totalDim
      )
    {
      let scale = sqrtf(2.0 / Float(kvLatentDim))
      for i in 0..<(kvLatentDim * totalDim) {
        kPtr[i] = Float16(Float.random(in: -scale...scale))
        vPtr[i] = Float16(Float.random(in: -scale...scale))
      }
    }
  }

  /// Load pre-trained decompression weights
  public func loadWeights(wk: MTLBuffer, wv: MTLBuffer) {
    wDecompressK = wk
    wDecompressV = wv
  }

  /// Execute GEMM using MFA's generated kernel
  /// C[M,N] = A[M,K] @ B[K,N]
  public func encodeGEMM(
    commandBuffer: MTLCommandBuffer,
    A: MTLBuffer,
    B: MTLBuffer,
    C: MTLBuffer,
    M: Int,
    N: Int,
    K: Int
  ) {
    // Create descriptor
    var desc = GEMMDescriptor()
    desc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
    desc.memoryPrecisions = (A: .FP16, B: .FP16, C: .FP16)
    desc.transposeState = (A: false, B: false)
    desc.loadPreviousC = false

    // Ensure kernel is registered (should be cached from init)
    GEMMKernel.register(descriptor: desc)

    // Retrieve cached kernel and pipeline
    guard let (kernel, pipeline) = GEMMKernel.pipelineCache[desc] else {
      fatalError("GEMM kernel not found in cache for \(M)×\(N)×\(K)")
    }

    // Encode command
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0
    )

    // Set buffers (MFA convention: 0=A, 1=B, 2=C)
    encoder.setBuffer(A, offset: 0, index: 0)
    encoder.setBuffer(B, offset: 0, index: 1)
    encoder.setBuffer(C, offset: 0, index: 2)

    // Calculate dispatch dimensions
    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }

    let gridSize = MTLSize(
      width: ceilDivide(N, kernel.blockDimensions.N),
      height: ceilDivide(M, kernel.blockDimensions.M),
      depth: 1
    )
    let groupSize = MTLSize(
      width: Int(kernel.threadgroupSize),
      height: 1,
      depth: 1
    )

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()
  }

  /// Forward pass: MFA GEMM decompression
  /// K[batch*seq, heads*dim] = latent_k[batch*seq, latent] @ W_k[latent, heads*dim]
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
    guard
      let wDecompressK,
      let wDecompressV
    else {
      throw NSError(
        domain: "MLAOptimizedGEMMMFA", code: 1,
        userInfo: [
          NSLocalizedDescriptionKey: "Weights not initialized",
        ]
      )
    }

    let totalDim = numHeads * headDim
    let kvSize = batchSize * sequenceLength * totalDim * MemoryLayout<Float16>.size

    // Allocate K,V buffers
    if self.decompressedK == nil || currentKVSize < kvSize {
      self.decompressedK = device.makeBuffer(
        length: kvSize, options: .storageModePrivate
      )
      self.decompressedV = device.makeBuffer(
        length: kvSize, options: .storageModePrivate
      )
      currentKVSize = kvSize
    }

    decompressedK = self.decompressedK
    decompressedV = self.decompressedV

    guard
      let kBuf = decompressedK,
      let vBuf = decompressedV
    else {
      throw NSError(
        domain: "MLAOptimizedGEMMMFA", code: 2,
        userInfo: [
          NSLocalizedDescriptionKey: "Failed to allocate buffers",
        ]
      )
    }

    // Batched GEMM for all batches at once
    // latent_k[batchSize*seqLen, kvLatentDim] @ W_k[kvLatentDim, totalDim]
    let M = batchSize * sequenceLength
    let N = totalDim
    let K = kvLatentDim

    // Decompress K
    encodeGEMM(
      commandBuffer: commandBuffer,
      A: kvLatent,
      B: wDecompressK,
      C: kBuf,
      M: M,
      N: N,
      K: K
    )

    // Decompress V
    encodeGEMM(
      commandBuffer: commandBuffer,
      A: kvLatent,
      B: wDecompressV,
      C: vBuf,
      M: M,
      N: N,
      K: K
    )
  }
}

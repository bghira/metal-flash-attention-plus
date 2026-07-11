//
//  HadamardRotation.swift
//  FlashAttention
//
//  Group-wise Hadamard rotation (ConvRot-style) for outlier smoothing.
//  Applies the Fast Walsh-Hadamard Transform (FWHT) to power-of-2 blocks
//  of a tensor, enabling lower-precision quantization with better accuracy.
//

import Metal

/// Applies group-wise Hadamard rotation to tensors on the GPU.
///
/// Inspired by ConvRot (arXiv:2512.03673), which uses Regular Hadamard
/// Transform to smooth outliers in diffusion transformer weights/activations,
/// enabling W4A4 quantization with negligible quality loss.
///
/// On Apple Silicon, the FWHT is particularly efficient:
/// - Only additions/subtractions (no multiplications)
/// - Butterfly pattern maps to simdgroup shared memory
/// - Unified memory eliminates copy overhead between rotation and attention
public final class HadamardRotation {
  private let device: MTLDevice
  private let commandQueue: MTLCommandQueue
  private var pipelineCache: [Int: MTLComputePipelineState] = [:]
  private let lock = NSLock()

  public init(device: MTLDevice? = nil) {
    self.device = device ?? MTLCreateSystemDefaultDevice()!
    self.commandQueue = self.device.makeCommandQueue()!
  }

  // MARK: - Public API

  /// Apply in-place Hadamard rotation to a buffer.
  ///
  /// - Parameters:
  ///   - buffer: The tensor buffer (FP32), shape [numBlocks, blockSize]
  ///   - blockSize: Size of each Hadamard block (must be power of 2, ≤ 1024)
  ///   - numBlocks: Number of independent blocks
  ///   - commandBuffer: Optional command buffer to encode into
  /// - Returns: Command buffer (committed if none provided)
  public func rotate(
    buffer: MTLBuffer,
    blockSize: Int,
    numBlocks: Int,
    commandBuffer: MTLCommandBuffer? = nil
  ) throws -> MTLCommandBuffer {
    precondition(blockSize > 0 && (blockSize & (blockSize - 1)) == 0,
                 "blockSize must be power of 2")
    precondition(blockSize <= 1024, "blockSize must be ≤ 1024")
    precondition(numBlocks > 0, "numBlocks must be > 0")

    let ownsCommandBuffer = commandBuffer == nil
    let cb = commandBuffer ?? commandQueue.makeCommandBuffer()!
    guard let encoder = cb.makeComputeCommandEncoder() else {
      throw NSError(domain: "HadamardRotation", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to create encoder"])
    }

    let pipeline = try getOrCreatePipeline(blockSize: blockSize)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)

    var blockSizeVal = UInt32(blockSize)
    var numBlocksVal = UInt32(numBlocks)
    encoder.setBytes(&blockSizeVal, length: 4, index: 1)
    encoder.setBytes(&numBlocksVal, length: 4, index: 2)

    // One thread per block (sequential FWHT within each block).
    let gridSize = MTLSize(width: numBlocks, height: 1, depth: 1)
    let threadSize = MTLSize(width: 1, height: 1, depth: 1)

    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadSize)
    encoder.endEncoding()

    if ownsCommandBuffer {
      cb.commit()
      cb.waitUntilCompleted()
    }

    return cb
  }

  /// Apply rotation to multiple buffers in a single command buffer.
  public func rotateBatch(
    buffers: [(buffer: MTLBuffer, blockSize: Int, numBlocks: Int)],
    commandBuffer: MTLCommandBuffer? = nil
  ) throws -> MTLCommandBuffer {
    let ownsCommandBuffer = commandBuffer == nil
    let cb = commandBuffer ?? commandQueue.makeCommandBuffer()!

    for item in buffers {
      _ = try rotate(
        buffer: item.buffer,
        blockSize: item.blockSize,
        numBlocks: item.numBlocks,
        commandBuffer: cb
      )
    }

    if ownsCommandBuffer {
      cb.commit()
      cb.waitUntilCompleted()
    }

    return cb
  }

  // MARK: - Kernel Source

  private static func kernelSource(blockSize: Int) -> String {
    let log2n = Int(log2(Double(blockSize)))

    return """
    #include <metal_stdlib>
    using namespace metal;

    kernel void hadamard_rotate(
        device float *data [[buffer(0)]],
        constant uint &block_size [[buffer(1)]],
        constant uint &num_blocks [[buffer(2)]],
        uint gid [[threadgroup_position_in_grid]])
    {
      if (gid >= num_blocks) return;

      device float *block = data + gid * block_size;

      // Fast Walsh-Hadamard Transform (in-place butterfly).
      // Each stage pairs elements 2^s apart.
      for (uint s = 0; s < \(log2n); s++) {
        uint stride = 1u << s;
        uint pair_dist = stride * 2;
        for (uint i = 0; i < block_size; i += pair_dist) {
          for (uint j = 0; j < stride; j++) {
            float a = block[i + j];
            float b = block[i + j + stride];
            block[i + j] = a + b;
            block[i + j + stride] = a - b;
          }
        }
      }

      // Normalize by 1/sqrt(N).
      float inv_sqrt_n = rsqrt((float)block_size);
      for (uint i = 0; i < block_size; i++) {
        block[i] *= inv_sqrt_n;
      }
    }
    """
  }

  // MARK: - Pipeline Management

  private func getOrCreatePipeline(blockSize: Int) throws -> MTLComputePipelineState {
    lock.lock()
    defer { lock.unlock() }

    if let cached = pipelineCache[blockSize] {
      return cached
    }

    let source = Self.kernelSource(blockSize: blockSize)
    let options = MTLCompileOptions()
    options.languageVersion = .version3_2

    let library = try device.makeLibrary(source: source, options: options)
    let function = try library.makeFunction(name: "hadamard_rotate")

    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = function
    pipelineDesc.maxTotalThreadsPerThreadgroup = 1024

    let pipeline = try device.makeComputePipelineState(
      descriptor: pipelineDesc, options: [], reflection: nil)

    pipelineCache[blockSize] = pipeline
    return pipeline
  }
}

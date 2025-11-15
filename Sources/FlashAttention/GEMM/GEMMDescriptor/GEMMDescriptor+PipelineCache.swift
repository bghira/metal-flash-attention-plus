//
//  GEMMDescriptor+PipelineCache.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Foundation
@preconcurrency import Metal

private extension NSLock {
  func withLock<T>(_ body: () throws -> T) rethrows -> T {
    lock()
    defer { unlock() }
    return try body()
  }
}

private final class GEMMKernelCache: @unchecked Sendable {
  static let shared = GEMMKernelCache()

  private let lock = NSLock()
  private var libraryCache: [GEMMKernelDescriptor: GEMMKernel.LibraryValue] = [:]
  private var pipelineCache: [GEMMDescriptor: GEMMKernel.PipelineValue] = [:]

  func cachedPipeline(for descriptor: GEMMDescriptor) -> GEMMKernel.PipelineValue? {
    lock.withLock {
      pipelineCache[descriptor]
    }
  }

  func storePipeline(_ value: GEMMKernel.PipelineValue, for descriptor: GEMMDescriptor) {
    lock.withLock {
      pipelineCache[descriptor] = value
    }
  }

  func cachedLibrary(
    for descriptor: GEMMKernelDescriptor
  ) -> GEMMKernel.LibraryValue? {
    lock.withLock {
      libraryCache[descriptor]
    }
  }

  func storeLibrary(
    _ value: GEMMKernel.LibraryValue,
    for descriptor: GEMMKernelDescriptor
  ) {
    lock.withLock {
      libraryCache[descriptor] = value
    }
  }
}

public extension GEMMKernel {
  typealias LibraryValue = (
    kernel: GEMMKernel, library: MTLLibrary
  )
  typealias PipelineValue = (
    kernel: GEMMKernel, pipeline: MTLComputePipelineState
  )

  static func cachedPipeline(for descriptor: GEMMDescriptor) -> PipelineValue? {
    GEMMKernelCache.shared.cachedPipeline(for: descriptor)
  }
}

public extension GEMMKernel {
  // Register this problem configuration in the cache.
  static func register(descriptor: GEMMDescriptor) {
    let cache = GEMMKernelCache.shared

    guard cache.cachedPipeline(for: descriptor) == nil else {
      return
    }

    var kernelDescriptor = GEMMKernelDescriptor(descriptor: descriptor)

    let device = MTLContext.global.device
    if device.supportsFamily(.apple9) {
      kernelDescriptor.preferAsyncStore = false
    } else {
      guard let blockDimensions = kernelDescriptor.blockDimensions else {
        fatalError("Block dimensions were not set.")
      }
      if blockDimensions == (48, 48, 32) {
        kernelDescriptor.preferAsyncStore = nil
      } else {
        kernelDescriptor.preferAsyncStore = true
      }
    }

    func createLibrary(_ kernelDescriptor: GEMMKernelDescriptor) -> LibraryValue {
      if let cached = cache.cachedLibrary(for: kernelDescriptor) {
        return cached
      }

      let kernel = GEMMKernel(descriptor: kernelDescriptor)
      let source = kernel.createSource()
      let library = try! device.makeLibrary(source: source, options: nil)

      let output = (kernel, library)
      cache.storeLibrary(output, for: kernelDescriptor)
      return output
    }

    func createPipeline(_ libraryValue: LibraryValue) -> PipelineValue {
      let constants = MTLFunctionConstantValues()
      descriptor.setFunctionConstants(constants)

      let library = libraryValue.library
      let function = try! library.makeFunction(
        name: "gemm", constantValues: constants
      )
      let pipeline = try! device.makeComputePipelineState(
        function: function
      )
      return (libraryValue.kernel, pipeline)
    }

    if kernelDescriptor.preferAsyncStore == nil {
      var candidates: [PipelineValue] = []
      for candidateID in 0..<4 {
        var blockDimensions: (M: UInt16, N: UInt16, K: UInt16)
        var preferAsyncStore: Bool
        switch candidateID {
        case 0:
          blockDimensions = (48, 48, 32)
          preferAsyncStore = false
        case 1:
          blockDimensions = (48, 48, 40)
          preferAsyncStore = false
        case 2:
          blockDimensions = (48, 48, 32)
          preferAsyncStore = true
        case 3:
          blockDimensions = (48, 48, 40)
          preferAsyncStore = true
        default:
          fatalError("This should never happen.")
        }

        // Set the attributes unique to this variant.
        var modifiedKernelDescriptor = kernelDescriptor
        modifiedKernelDescriptor.blockDimensions = blockDimensions
        modifiedKernelDescriptor.preferAsyncStore = preferAsyncStore

        let libraryValue = createLibrary(modifiedKernelDescriptor)
        let pipelineValue = createPipeline(libraryValue)
        candidates.append(pipelineValue)
      }

      // Find the maximum occupancy.
      var maximumOccupancy: Int = -1
      for candidate in candidates {
        let pipeline = candidate.pipeline
        let occupancy = pipeline.maxTotalThreadsPerThreadgroup
        maximumOccupancy = max(maximumOccupancy, occupancy)
      }
      candidates.removeAll(where: {
        $0.pipeline.maxTotalThreadsPerThreadgroup != maximumOccupancy
      })

      // Choose the highest-performing candidate.
      if let selected = candidates.last {
        cache.storePipeline(selected, for: descriptor)
      }
    } else {
      let libraryValue = createLibrary(kernelDescriptor)
      let pipelineValue = createPipeline(libraryValue)
      cache.storePipeline(pipelineValue, for: descriptor)
    }
  }
}

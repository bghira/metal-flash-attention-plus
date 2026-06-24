//
//  GEMMDeviceCapabilities.swift
//  FlashAttention
//
//  Created on 2025-09-17
//

import Metal

/// Fallback strategies for quantization when device capabilities are insufficient
public enum QuantizationFallbackStrategy: Equatable {
  /// Use blockwise quantization with symmetric strategy for specified operands
  case symmetricBlockwise(weights: Bool, activations: Bool)

  /// Fall back to asymmetric per-channel quantization
  case asymmetricPerChannel

  /// Fall back to FP16 GEMM without quantization
  case fp16GEMM
}

/// Result of quantization strategy determination
public struct QuantizationStrategyResult {
  public let mode: QuantizationMode
  public let warnings: [String]

  public init(mode: QuantizationMode, warnings: [String] = []) {
    self.mode = mode
    self.warnings = warnings
  }
}

/// Extension to MTLDevice providing device capability checks for blockwise quantization
public extension MTLDevice {
  /// Check if device supports blockwise quantization operations
  /// Requires simdgroup operations, subgroup reductions, and appropriate matrix support
  var supportsBlockwiseQuantization: Bool {
    // Blockwise quantization requires:
    // 1. Simdgroup matrix operations (Apple7+)
    // 2. Async copy operations for efficient data movement
    // 3. Sufficient simdgroup size for matrix tiling

    guard supportsFamily(.apple7) else {
      return false
    }

    // Check for minimum simdgroup size (32 threads for efficient matrix operations)
    guard simdgroupSize >= 32 else {
      return false
    }

    // Additional checks can be added here for specific quantization requirements
    return true
  }

  /// Check if device has native BFloat16 support
  /// Requires Apple9 family and compatible OS version
  var supportsBFloat16: Bool {
    // BFloat16 support requires Apple9 family (M3+) and macOS 15.0+
    guard supportsFamily(.apple9) else {
      return false
    }

    // Additional OS version check for BF16 availability
    if #available(macOS 15.0, iOS 17.0, *) {
      return true
    } else {
      return false
    }
  }

  /// Query runtime simdgroup size (typically 32 or 64)
  var simdgroupSize: Int {
    // On Apple Silicon, simdgroups are typically 32 threads
    // This can be queried more precisely through Metal feature sets

    if supportsFamily(.apple7) {
      // Apple7+ typically has 32-thread simdgroups
      32
    } else {
      // Older or non-Apple GPUs may have different sizes
      32 // Conservative default
    }
  }

  /// Check if device supports simdgroup matrix operations
  var supportsSimdgroupMatrixOps: Bool {
    supportsFamily(.apple7)
  }

  /// Check if device supports simdgroup permute operations
  var supportsSimdgroupPermute: Bool {
    supportsFamily(.apple7)
  }

  /// Check if device supports async simdgroup copy operations
  var supportsAsyncSimdgroupCopy: Bool {
    supportsFamily(.apple7)
  }

  /// Determine the best available quantization strategy for a requested mode
  /// - Parameter requested: The desired quantization mode
  /// - Returns: Result containing the actual mode to use and any warnings
  func quantizationStrategy(requested: QuantizationMode) -> QuantizationStrategyResult {
    var warnings: [String] = []

    switch requested {
    case let .blockwise(blockSizeK, bothOperands):
      // Check if device supports blockwise quantization
      if !supportsBlockwiseQuantization {
        warnings
          .append(
            "Device does not support efficient blockwise quantization (requires Apple7+ with simdgroup matrix operations)"
          )

        if supportsBFloat16 {
          warnings
            .append(
              "Falling back to BFloat16 GEMM for better performance than INT8 without hardware acceleration"
            )
          return QuantizationStrategyResult(mode: .tensorWise, warnings: warnings)
        } else {
          warnings.append("Falling back to tensor-wise INT8 quantization")
          return QuantizationStrategyResult(mode: .tensorWise, warnings: warnings)
        }
      }

      // Check block size constraints
      if blockSizeK % 8 != 0 {
        warnings
          .append(
            "Block size \(blockSizeK) is not a multiple of 8; rounding to \(((blockSizeK + 7) / 8) * 8)"
          )
        let adjustedBlockSize = ((blockSizeK + 7) / 8) * 8
        return QuantizationStrategyResult(
          mode: .blockwise(blockSizeK: adjustedBlockSize, bothOperands: bothOperands),
          warnings: warnings
        )
      }

      // Check if simdgroup size is sufficient for the block size
      if blockSizeK > simdgroupSize * 4 {
        warnings
          .append(
            "Block size \(blockSizeK) may be too large for optimal performance (simdgroup size: \(simdgroupSize))"
          )
        warnings
          .append(
            "Consider using block size ≤ \(simdgroupSize * 4) for better cache utilization"
          )
      }

      // Blockwise quantization is supported
      return QuantizationStrategyResult(mode: requested, warnings: warnings)

    case .rowWise:
      // Row-wise quantization has fewer hardware requirements
      if !supportsSimdgroupMatrixOps {
        warnings
          .append(
            "Device lacks simdgroup matrix operations; row-wise quantization may have reduced performance"
          )
        warnings.append("Consider using tensor-wise quantization for better compatibility")
      }

      return QuantizationStrategyResult(mode: requested, warnings: warnings)

    case .tensorWise:
      // Tensor-wise quantization is supported on all devices
      if !supportsSimdgroupMatrixOps {
        warnings
          .append("Device lacks advanced matrix operations; performance may be limited")
      }

      return QuantizationStrategyResult(mode: requested, warnings: warnings)
    }
  }

  /// Get the optimal fallback strategy when quantization is not feasible
  /// - Parameter originalMode: The originally requested quantization mode
  /// - Returns: Recommended fallback strategy
  func getQuantizationFallbackStrategy(for originalMode: QuantizationMode)
    -> QuantizationFallbackStrategy
  {
    switch originalMode {
    case let .blockwise(_, bothOperands):
      if supportsSimdgroupMatrixOps {
        // Can still do some blockwise operations, just not fully optimized
        .symmetricBlockwise(weights: true, activations: bothOperands)
      } else if supportsBFloat16 {
        // BFloat16 often provides better accuracy than INT8 without hardware acceleration
        .fp16GEMM
      } else {
        // Fall back to per-channel quantization
        .asymmetricPerChannel
      }

    case .rowWise:
      if supportsBFloat16 {
        .fp16GEMM
      } else {
        .asymmetricPerChannel
      }

    case .tensorWise:
      // For tensor-wise, per-channel is a reasonable step-up
      .asymmetricPerChannel
    }
  }

  /// Generate detailed capability report for debugging
  var quantizationCapabilityReport: String {
    var report = "=== Metal Device Quantization Capabilities ===\n"
    report += "Device: \(name)\n"
    report += "Supports Apple7+: \(supportsFamily(.apple7))\n"
    report += "Supports Apple9+: \(supportsFamily(.apple9))\n"
    report += "Simdgroup Size: \(simdgroupSize)\n"
    report += "Supports Blockwise Quantization: \(supportsBlockwiseQuantization)\n"
    report += "Supports BFloat16: \(supportsBFloat16)\n"
    report += "Supports Simdgroup Matrix Ops: \(supportsSimdgroupMatrixOps)\n"
    report += "Supports Simdgroup Permute: \(supportsSimdgroupPermute)\n"
    report += "Supports Async Simdgroup Copy: \(supportsAsyncSimdgroupCopy)\n"
    report += "\nRecommended Strategies:\n"

    if supportsBlockwiseQuantization {
      report += "✅ Blockwise quantization with block sizes: 32, 64, 128, 256\n"
    } else {
      report += "❌ Blockwise quantization not recommended\n"
    }

    if supportsBFloat16 {
      report += "✅ BFloat16 GEMM for high accuracy\n"
    } else {
      report += "⚠️  BFloat16 not available, use FP16 instead\n"
    }

    if supportsSimdgroupMatrixOps {
      report += "✅ Row-wise and tensor-wise quantization supported\n"
    } else {
      report += "⚠️  Limited quantization support without simdgroup matrix operations\n"
    }

    return report
  }
}

/// Utility methods for quantization strategy selection
public extension QuantizationMode {
  /// Get a device-optimized version of this quantization mode
  /// - Parameter device: Target Metal device
  /// - Returns: Optimized quantization mode with warnings
  func optimizedForDevice(_ device: MTLDevice) -> QuantizationStrategyResult {
    device.quantizationStrategy(requested: self)
  }

  /// Check if this quantization mode is efficiently supported on the device
  /// - Parameter device: Target Metal device
  /// - Returns: True if the mode is well-supported, false if fallbacks are recommended
  func isEfficientlySupported(on device: MTLDevice) -> Bool {
    let result = device.quantizationStrategy(requested: self)
    return result.warnings.isEmpty && isEqual(to: result.mode)
  }

  /// Compare two QuantizationMode values for equality
  /// - Parameter other: The other mode to compare with
  /// - Returns: True if the modes are equivalent
  private func isEqual(to other: QuantizationMode) -> Bool {
    switch (self, other) {
    case (.tensorWise, .tensorWise):
      true
    case (.rowWise, .rowWise):
      true
    case let (.blockwise(size1, both1), .blockwise(size2, both2)):
      size1 == size2 && both1 == both2
    default:
      false
    }
  }
}

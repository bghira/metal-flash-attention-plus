//
//  GEMMOperandPrecision.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

/// An enumeration of the precisions supported by the kernel.
///
/// This implementation supports quantized precisions optimized for Apple GPU
/// matrix acceleration. The quantized formats follow
/// these design principles:
/// - Keep data compressed in `device` or `threadgroup` memory
/// - Transform to floating point when loading into registers
/// - Keep accumulator in floating point until output needs to be written
/// - Compress when writing back to device memory
///
/// GPU-accelerated quantization formats:
/// - INT8: 8-bit signed integers with 16-bit or 32-bit accumulation
/// - INT4: 4-bit integers using optimized GPU instructions (16 levels)
/// - Scaling factors and zero points stored separately for dequantization
public enum GEMMOperandPrecision: UInt16 {
  case FP32 = 0
  case FP16 = 1
  case BF16 = 2
  case INT8 = 3
  case INT4 = 4

  // The MSL keyword corresponding to the precision.
  public var name: String {
    switch self {
    case .FP32:
      "float"
    case .FP16:
      "half"
    case .BF16:
      "bfloat"
    case .INT8:
      "char"
    case .INT4:
      "uchar" // Stored as packed 4-bit values in 8-bit containers
    }
  }

  // The size of a scalar, in bytes.
  public var size: Int {
    switch self {
    case .FP32:
      4
    case .FP16:
      2
    case .BF16:
      2
    case .INT8:
      1
    case .INT4:
      1 // Two 4-bit values packed into one byte
    }
  }

  /// Whether this precision requires quantization parameters (scale and zero point)
  public var requiresQuantizationParameters: Bool {
    switch self {
    case .FP32, .FP16, .BF16:
      false
    case .INT8, .INT4:
      true
    }
  }

  /// The accumulator precision to use for this operand precision
  public var accumulatorPrecision: GEMMOperandPrecision {
    switch self {
    case .FP32, .FP16, .BF16:
      .FP32
    case .INT8:
      .FP32 // Use FP32 accumulator for INT8 operations
    case .INT4:
      .FP32 // Use FP32 accumulator for INT4 operations
    }
  }
}

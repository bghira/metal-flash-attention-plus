//
//  GEMMQuantization.swift
//  FlashAttention
//
//

import Metal

/// Quantization parameters for tensor quantization
public struct QuantizationParameters {
    /// Scale factor for dequantization: dequantized_value = (quantized_value - zero_point) * scale
    public var scale: Float

    /// Zero point for quantization (subtracted before scaling)
    public var zeroPoint: Int32

    /// The precision of the quantized data
    public var precision: GEMMOperandPrecision

    public init(scale: Float, zeroPoint: Int32, precision: GEMMOperandPrecision) {
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.precision = precision
    }
}

/// Extension to handle quantization operations
extension GEMMOperandPrecision {
    /// Calculate quantization parameters for a tensor
    /// - Parameters:
    ///   - data: Input floating point data
    ///   - count: Number of elements
    /// - Returns: Quantization parameters optimized for the precision
    public func calculateQuantizationParameters(data: UnsafePointer<Float>, count: Int) -> QuantizationParameters {
        // Find min and max values
        var minVal = Float.greatestFiniteMagnitude
        var maxVal = -Float.greatestFiniteMagnitude

        for i in 0..<count {
            let val = data[i]
            minVal = min(minVal, val)
            maxVal = max(maxVal, val)
        }

        switch self {
        case .INT8:
            // Symmetric quantization for INT8 (-128 to 127)
            let absMax = max(abs(minVal), abs(maxVal))
            let scale = absMax / 127.0
            return QuantizationParameters(scale: scale, zeroPoint: 0, precision: .INT8)

        case .INT4:
            // Symmetric quantization for INT4 (-8 to 7, stored in 4-bit)
            let absMax = max(abs(minVal), abs(maxVal))
            let scale = absMax / 7.0
            return QuantizationParameters(scale: scale, zeroPoint: 0, precision: .INT4)

        default:
            fatalError("Quantization parameters only supported for INT8 and INT4")
        }
    }

    /// Quantize floating point data to the specified precision
    /// - Parameters:
    ///   - input: Input floating point data
    ///   - output: Output buffer for quantized data
    ///   - count: Number of elements
    ///   - parameters: Quantization parameters
    public func quantize(input: UnsafePointer<Float>,
                        output: UnsafeMutableRawPointer,
                        count: Int,
                        parameters: QuantizationParameters) {
        switch self {
        case .INT8:
            let outputInt8 = output.bindMemory(to: Int8.self, capacity: count)
            for i in 0..<count {
                let quantized = Int32(round(input[i] / parameters.scale)) + parameters.zeroPoint
                outputInt8[i] = Int8(clamping: quantized)
            }

        case .INT4:
            let outputUInt8 = output.bindMemory(to: UInt8.self, capacity: (count + 1) / 2)
            for i in stride(from: 0, to: count, by: 2) {
                // Pack two 4-bit values into one byte
                let val1 = Int32(round(input[i] / parameters.scale)) + parameters.zeroPoint
                let val2 = i + 1 < count ? Int32(round(input[i + 1] / parameters.scale)) + parameters.zeroPoint : 0

                let packed1 = UInt8(clamping: val1 + 8) & 0xF  // Convert from [-8,7] to [0,15]
                let packed2 = UInt8(clamping: val2 + 8) & 0xF

                outputUInt8[i / 2] = (packed2 << 4) | packed1
            }

        default:
            fatalError("Quantization only supported for INT8 and INT4")
        }
    }

    /// Dequantize data back to floating point
    /// - Parameters:
    ///   - input: Quantized input data
    ///   - output: Output floating point buffer
    ///   - count: Number of elements
    ///   - parameters: Quantization parameters
    public func dequantize(input: UnsafeRawPointer,
                          output: UnsafeMutablePointer<Float>,
                          count: Int,
                          parameters: QuantizationParameters) {
        switch self {
        case .INT8:
            let inputInt8 = input.bindMemory(to: Int8.self, capacity: count)
            for i in 0..<count {
                output[i] = (Float(inputInt8[i]) - Float(parameters.zeroPoint)) * parameters.scale
            }

        case .INT4:
            let inputUInt8 = input.bindMemory(to: UInt8.self, capacity: (count + 1) / 2)
            for i in stride(from: 0, to: count, by: 2) {
                let packed = inputUInt8[i / 2]
                let val1 = Int32(packed & 0xF) - 8  // Convert from [0,15] to [-8,7]
                let val2 = Int32(packed >> 4) - 8

                output[i] = (Float(val1) - Float(parameters.zeroPoint)) * parameters.scale
                if i + 1 < count {
                    output[i + 1] = (Float(val2) - Float(parameters.zeroPoint)) * parameters.scale
                }
            }

        default:
            fatalError("Dequantization only supported for INT8 and INT4")
        }
    }
}

/// Utility class for managing quantized tensor operations
public class QuantizedTensor {
    public let data: MTLBuffer
    public let parameters: QuantizationParameters
    public let elementCount: Int
    public let originalShape: [Int]

    public init(device: MTLDevice,
                data: MTLBuffer,
                parameters: QuantizationParameters,
                elementCount: Int,
                shape: [Int]) {
        self.data = data
        self.parameters = parameters
        self.elementCount = elementCount
        self.originalShape = shape
    }

    /// Create a quantized tensor from floating point data
    /// - Parameters:
    ///   - device: Metal device
    ///   - floatData: Input floating point data
    ///   - shape: Tensor shape
    ///   - precision: Target quantization precision
    /// - Returns: Quantized tensor
    public static func from(device: MTLDevice,
                           floatData: [Float],
                           shape: [Int],
                           precision: GEMMOperandPrecision) -> QuantizedTensor {
        let elementCount = floatData.count

        let parameters: QuantizationParameters
        let bufferSize: Int

        if precision.requiresQuantizationParameters {
            parameters = precision.calculateQuantizationParameters(data: floatData, count: elementCount)
            bufferSize = precision == .INT4 ? (elementCount + 1) / 2 : elementCount * precision.size
        } else {
            // For non-quantized types (FP32, FP16, BF16), create dummy parameters
            parameters = QuantizationParameters(scale: 1.0, zeroPoint: 0, precision: precision)
            bufferSize = elementCount * precision.size
        }

        guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            fatalError("Could not create quantized buffer")
        }

        floatData.withUnsafeBufferPointer { floatPtr in
            if precision.requiresQuantizationParameters {
                precision.quantize(input: floatPtr.baseAddress!,
                                 output: buffer.contents(),
                                 count: elementCount,
                                 parameters: parameters)
            } else {
                // For non-quantized types, just copy the data in the appropriate format
                switch precision {
                case .FP32:
                    let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
                    for i in 0..<elementCount {
                        pointer[i] = floatData[i]
                    }
                case .FP16:
                    let pointer = buffer.contents().assumingMemoryBound(to: Float16.self)
                    for i in 0..<elementCount {
                        pointer[i] = Float16(floatData[i])
                    }
                case .BF16:
                    let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
                    for i in 0..<elementCount {
                        let value32 = floatData[i].bitPattern
                        let value16 = unsafeBitCast(value32, to: SIMD2<UInt16>.self)[1]
                        pointer[i] = value16
                    }
                default:
                    break
                }
            }
        }

        return QuantizedTensor(device: device,
                             data: buffer,
                             parameters: parameters,
                             elementCount: elementCount,
                             shape: shape)
    }

    /// Convert quantized tensor back to floating point
    /// - Returns: Array of floating point values
    public func toFloats() -> [Float] {
        var result = Array<Float>(repeating: 0, count: elementCount)
        result.withUnsafeMutableBufferPointer { floatPtr in
            parameters.precision.dequantize(input: data.contents(),
                                          output: floatPtr.baseAddress!,
                                          count: elementCount,
                                          parameters: parameters)
        }
        return result
    }
}
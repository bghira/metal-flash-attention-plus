//
//  GEMMQuantization.swift
//  FlashAttention
//
//

@preconcurrency import Metal

private enum QuantizedTensorCoding {
  static let deviceKey: CodingUserInfoKey = {
    guard let key = CodingUserInfoKey(rawValue: "MFAQuantizedTensorDevice") else {
      fatalError("Failed to create coding key for quantized tensor device.")
    }
    return key
  }()

  final class DeviceBox: @unchecked Sendable {
    let device: MTLDevice

    init(device: MTLDevice) {
      self.device = device
    }
  }
}

/// Quantization mode specifying the granularity of quantization
public enum QuantizationMode: Codable {
  /// Quantize entire tensor with single scale/zero-point pair
  case tensorWise

  /// Quantize per block along K dimension for better accuracy
  /// - Parameters:
  ///   - blockSizeK: Size of blocks along K dimension (must be multiple of 8)
  ///   - bothOperands: If true, apply blockwise to both A and B operands; if false, only to weights
  case blockwise(blockSizeK: Int, bothOperands: Bool = false)

  /// Quantize per row for preserving fine-grained distributions
  case rowWise

  /// Default block size for block-wise quantization along K dimension
  public static let defaultBlockSizeK = 128
}

/// Quantization strategy used to interpret scale/zero-point pairs.
public enum QuantizationStrategy: UInt8, Codable, CaseIterable {
  /// Matches the original quantization behavior (per-value affine transforms).
  case legacy = 0
  /// Allows non-zero symmetric ranges (per-block asymmetric affine quantization).
  case asymmetric = 1
  /// Enforces zero-point free symmetric quantization with block centering.
  case symmetric = 2

  /// Latest strategy encoding version; bump when serialized layout changes.
  public static let currentVersion: UInt8 = 1
}

extension QuantizationMode {
  private enum CodingKeys: String, CodingKey {
    case caseName
    case blockSize
    case bothOperands
  }

  private enum CaseName: String, Codable {
    case tensorWise
    case blockwise
    case rowWise
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .tensorWise:
      try container.encode(CaseName.tensorWise, forKey: .caseName)
    case let .blockwise(blockSizeK, bothOperands):
      try container.encode(CaseName.blockwise, forKey: .caseName)
      try container.encode(blockSizeK, forKey: .blockSize)
      try container.encode(bothOperands, forKey: .bothOperands)
    case .rowWise:
      try container.encode(CaseName.rowWise, forKey: .caseName)
    }
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let caseName = try container.decode(CaseName.self, forKey: .caseName)
    switch caseName {
    case .tensorWise:
      self = .tensorWise
    case .blockwise:
      let blockSizeK = try container.decode(Int.self, forKey: .blockSize)
      let bothOperands = try container.decodeIfPresent(Bool.self, forKey: .bothOperands) ?? false
      self = .blockwise(blockSizeK: blockSizeK, bothOperands: bothOperands)
    case .rowWise:
      self = .rowWise
    }
  }
}

/// Quantization parameters for tensor quantization
public struct QuantizationParameters: Codable {
  public static let currentStrategyVersion = QuantizationStrategy.currentVersion

  /// Scale factor for dequantization: dequantized_value = (quantized_value - zero_point) * scale
  /// For non-uniform quantization modes, this can be an array of scales
  public var scale: Float

  /// Zero point for quantization (subtracted before scaling)
  public var zeroPoint: Int32

  /// The precision of the quantized data
  public var precision: GEMMOperandPrecision

  /// Quantization mode (tensor-wise, block-wise, row-wise)
  public var mode: QuantizationMode

  /// Additional scales for block-wise or row-wise quantization
  public var additionalScales: [Float]?

  /// Additional zero points for block-wise or row-wise quantization
  public var additionalZeroPoints: [Int32]?

  /// Strategy describing how scale/zero-point pairs should be interpreted.
  public var strategy: QuantizationStrategy

  /// Version number for serialized strategy payloads.
  public var strategyVersion: UInt8

  public init(
    scale: Float,
    zeroPoint: Int32,
    precision: GEMMOperandPrecision,
    mode: QuantizationMode = .tensorWise,
    strategy: QuantizationStrategy = .legacy,
    strategyVersion: UInt8 = QuantizationParameters.currentStrategyVersion
  ) {
    self.scale = scale
    self.zeroPoint = zeroPoint
    self.precision = precision
    self.mode = mode
    additionalScales = nil
    additionalZeroPoints = nil
    self.strategy = strategy
    self.strategyVersion = strategyVersion

    QuantizationParameters.validate(
      strategy: strategy,
      zeroPoints: [zeroPoint],
      precision: precision,
      mode: mode
    )
  }

  /// Initialize with multiple scales and zero points for non-uniform quantization
  public init(
    scales: [Float],
    zeroPoints: [Int32],
    precision: GEMMOperandPrecision,
    mode: QuantizationMode,
    strategy: QuantizationStrategy = .legacy,
    strategyVersion: UInt8 = QuantizationParameters.currentStrategyVersion
  ) {
    scale = scales.first ?? 1.0
    zeroPoint = zeroPoints.first ?? 0
    self.precision = precision
    self.mode = mode
    additionalScales = scales.count > 1 ? Array(scales.dropFirst()) : nil
    additionalZeroPoints = zeroPoints.count > 1 ? Array(zeroPoints.dropFirst()) : nil
    self.strategy = strategy
    self.strategyVersion = strategyVersion

    let allZeroPoints = zeroPoints.isEmpty ? [zeroPoint] : zeroPoints
    QuantizationParameters.validate(
      strategy: strategy,
      zeroPoints: allZeroPoints,
      precision: precision,
      mode: mode
    )
  }

  private static func validate(
    strategy: QuantizationStrategy,
    zeroPoints: [Int32],
    precision: GEMMOperandPrecision,
    mode: QuantizationMode
  ) {
    guard precision.requiresQuantizationParameters else {
      if strategy != .legacy {
        print(
          "Warning: Ignoring \(strategy) strategy for precision \(precision); quantization parameters are unused."
        )
      }
      return
    }

    guard strategy == .symmetric else { return }

    if let nonZero = zeroPoints.first(where: { $0 != 0 }) {
      preconditionFailure(
        "Symmetric quantization requires zero points to be zero; found \(nonZero)."
      )
    }

    if case let .blockwise(blockSize, _) = mode {
      precondition(
        blockSize % 8 == 0,
        "Symmetric block-wise quantization requires block sizes that are multiples of 8."
      )
    }
  }

  private enum CodingKeys: String, CodingKey {
    case scale
    case zeroPoint
    case precision
    case mode
    case additionalScales
    case additionalZeroPoints
    case strategy
    case strategyVersion
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    scale = try container.decode(Float.self, forKey: .scale)
    zeroPoint = try container.decode(Int32.self, forKey: .zeroPoint)
    precision = try container.decode(GEMMOperandPrecision.self, forKey: .precision)
    mode = try container.decode(QuantizationMode.self, forKey: .mode)
    additionalScales = try container.decodeIfPresent([Float].self, forKey: .additionalScales)
    additionalZeroPoints = try container.decodeIfPresent(
      [Int32].self,
      forKey: .additionalZeroPoints
    )
    strategy = try container
      .decodeIfPresent(QuantizationStrategy.self, forKey: .strategy) ?? .legacy
    strategyVersion = try container
      .decodeIfPresent(UInt8.self, forKey: .strategyVersion) ?? QuantizationParameters
      .currentStrategyVersion

    let stackedZeroPoints: [Int32] = if let extras = additionalZeroPoints, !extras.isEmpty {
      [zeroPoint] + extras
    } else {
      [zeroPoint]
    }

    QuantizationParameters.validate(
      strategy: strategy,
      zeroPoints: stackedZeroPoints,
      precision: precision,
      mode: mode
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(scale, forKey: .scale)
    try container.encode(zeroPoint, forKey: .zeroPoint)
    try container.encode(precision, forKey: .precision)
    try container.encode(mode, forKey: .mode)
    try container.encodeIfPresent(additionalScales, forKey: .additionalScales)
    try container.encodeIfPresent(additionalZeroPoints, forKey: .additionalZeroPoints)
    try container.encode(strategy, forKey: .strategy)
    try container.encode(strategyVersion, forKey: .strategyVersion)
  }
}

/// Extension to handle quantization operations
public extension GEMMOperandPrecision {
  /// Calculate quantization parameters for a tensor
  /// - Parameters:
  ///   - data: Input floating point data
  ///   - count: Number of elements
  ///   - shape: Tensor shape for block-wise or row-wise quantization
  ///   - mode: Quantization mode (tensor-wise, block-wise, row-wise)
  /// - Returns: Quantization parameters optimized for the precision
  func calculateQuantizationParameters(
    data: UnsafePointer<Float>,
    count: Int,
    shape: [Int] = [],
    mode: QuantizationMode = .tensorWise,
    strategy: QuantizationStrategy = .legacy
  )
    -> QuantizationParameters
  {
    switch mode {
    case .tensorWise:
      calculateTensorWiseParameters(data: data, count: count, strategy: strategy)

    case let .blockwise(blockSize, _):
      calculateBlockWiseParameters(
        data: data,
        count: count,
        shape: shape,
        blockSize: blockSize,
        strategy: strategy
      )

    case .rowWise:
      calculateRowWiseParameters(data: data, count: count, shape: shape, strategy: strategy)
    }
  }

  /// Calculate tensor-wise quantization parameters
  private func calculateTensorWiseParameters(
    data: UnsafePointer<Float>,
    count: Int,
    strategy: QuantizationStrategy
  )
    -> QuantizationParameters
  {
    // Find min and max values across entire tensor
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
      return QuantizationParameters(
        scale: scale,
        zeroPoint: 0,
        precision: .INT8,
        mode: .tensorWise,
        strategy: strategy
      )

    case .INT4:
      // Symmetric quantization for INT4 (-8 to 7, stored in 4-bit)
      let absMax = max(abs(minVal), abs(maxVal))
      let scale = absMax / 7.0
      return QuantizationParameters(
        scale: scale,
        zeroPoint: 0,
        precision: .INT4,
        mode: .tensorWise,
        strategy: strategy
      )

    default:
      fatalError("Quantization parameters only supported for INT8 and INT4")
    }
  }

  /// Calculate block-wise quantization parameters for better accuracy
  private func calculateBlockWiseParameters(
    data: UnsafePointer<Float>,
    count: Int,
    shape: [Int],
    blockSize: Int,
    strategy: QuantizationStrategy
  )
    -> QuantizationParameters
  {
    guard shape.count >= 2 else {
      // Fall back to tensor-wise if shape is insufficient
      return calculateTensorWiseParameters(data: data, count: count, strategy: strategy)
    }

    let rows = shape[0]
    let cols = shape[1]
    let numBlocksRow = (rows + blockSize - 1) / blockSize
    let numBlocksCol = (cols + blockSize - 1) / blockSize

    var scales: [Float] = []
    var zeroPoints: [Int32] = []

    for blockRow in 0..<numBlocksRow {
      for blockCol in 0..<numBlocksCol {
        let startRow = blockRow * blockSize
        let endRow = min(startRow + blockSize, rows)
        let startCol = blockCol * blockSize
        let endCol = min(startCol + blockSize, cols)

        var minVal = Float.greatestFiniteMagnitude
        var maxVal = -Float.greatestFiniteMagnitude

        // Find min/max within this block
        for r in startRow..<endRow {
          for c in startCol..<endCol {
            let idx = r * cols + c
            if idx < count {
              let val = data[idx]
              minVal = min(minVal, val)
              maxVal = max(maxVal, val)
            }
          }
        }

        // Calculate scale and zero point for this block
        let absMax = max(abs(minVal), abs(maxVal))
        let scale: Float
        switch self {
        case .INT8:
          scale = absMax / 127.0
        case .INT4:
          scale = absMax / 7.0
        default:
          fatalError("Block-wise quantization only supported for INT8 and INT4")
        }

        scales.append(scale)
        zeroPoints.append(0) // Using symmetric quantization
      }
    }

    return QuantizationParameters(
      scales: scales,
      zeroPoints: zeroPoints,
      precision: self,
      mode: .blockwise(blockSizeK: blockSize, bothOperands: false),
      strategy: strategy
    )
  }

  /// Calculate row-wise quantization parameters
  private func calculateRowWiseParameters(
    data: UnsafePointer<Float>,
    count: Int,
    shape: [Int],
    strategy: QuantizationStrategy
  )
    -> QuantizationParameters
  {
    guard shape.count >= 2 else {
      // Fall back to tensor-wise if shape is insufficient
      return calculateTensorWiseParameters(data: data, count: count, strategy: strategy)
    }

    let rows = shape[0]
    let cols = shape[1]
    var scales: [Float] = []
    var zeroPoints: [Int32] = []

    for row in 0..<rows {
      var minVal = Float.greatestFiniteMagnitude
      var maxVal = -Float.greatestFiniteMagnitude

      // Find min/max within this row
      for col in 0..<cols {
        let idx = row * cols + col
        if idx < count {
          let val = data[idx]
          minVal = min(minVal, val)
          maxVal = max(maxVal, val)
        }
      }

      // Calculate scale and zero point for this row
      let absMax = max(abs(minVal), abs(maxVal))
      let scale: Float
      switch self {
      case .INT8:
        scale = absMax / 127.0
      case .INT4:
        scale = absMax / 7.0
      default:
        fatalError("Row-wise quantization only supported for INT8 and INT4")
      }

      scales.append(scale)
      zeroPoints.append(0) // Using symmetric quantization
    }

    return QuantizationParameters(
      scales: scales,
      zeroPoints: zeroPoints,
      precision: self,
      mode: .rowWise,
      strategy: strategy
    )
  }

  /// Quantize floating point data to the specified precision
  /// - Parameters:
  ///   - input: Input floating point data
  ///   - output: Output buffer for quantized data
  ///   - count: Number of elements
  ///   - parameters: Quantization parameters
  func quantize(
    input: UnsafePointer<Float>,
    output: UnsafeMutableRawPointer,
    count: Int,
    parameters: QuantizationParameters
  ) {
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
        let val2 =
          i + 1 < count ? Int32(round(input[i + 1] / parameters.scale)) + parameters.zeroPoint : 0

        let packed1 = UInt8(max(
          0,
          min(15, val1 + 8)
        )) // Clamp [-8,7] to [0,15] before UInt8 conversion
        let packed2 = UInt8(max(0, min(15, val2 + 8)))

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
  func dequantize(
    input: UnsafeRawPointer,
    output: UnsafeMutablePointer<Float>,
    count: Int,
    parameters: QuantizationParameters
  ) {
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
        let val1 = Int32(packed & 0xF) - 8 // Convert from [0,15] to [-8,7]
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
public class QuantizedTensor: Codable {
  public let data: MTLBuffer
  public let parameters: QuantizationParameters
  public let elementCount: Int
  public let originalShape: [Int]

  // Block-wise quantization buffers
  public let blockScales: MTLBuffer?
  public let blockZeroPoints: MTLBuffer?
  public let blockSizeK: Int?
  public let precomputedSums: MTLBuffer?

  public init(
    device _: MTLDevice,
    data: MTLBuffer,
    parameters: QuantizationParameters,
    elementCount: Int,
    shape: [Int],
    blockScales: MTLBuffer? = nil,
    blockZeroPoints: MTLBuffer? = nil,
    blockSizeK: Int? = nil,
    precomputedSums: MTLBuffer? = nil
  ) {
    self.data = data
    self.parameters = parameters
    self.elementCount = elementCount
    originalShape = shape
    self.blockScales = blockScales
    self.blockZeroPoints = blockZeroPoints
    self.blockSizeK = blockSizeK
    self.precomputedSums = precomputedSums
  }

  /// Create a quantized tensor from floating point data
  /// - Parameters:
  ///   - device: Metal device
  ///   - floatData: Input floating point data
  ///   - shape: Tensor shape
  ///   - precision: Target quantization precision
  /// - Returns: Quantized tensor
  public static func from(
    device: MTLDevice,
    floatData: [Float],
    shape: [Int],
    precision: GEMMOperandPrecision,
    mode: QuantizationMode = .tensorWise,
    strategy: QuantizationStrategy = .legacy,
    blockScales: MTLBuffer? = nil,
    blockZeroPoints: MTLBuffer? = nil,
    precomputedSums: MTLBuffer? = nil
  )
    -> QuantizedTensor
  {
    let elementCount = floatData.count

    let parameters: QuantizationParameters
    let bufferSize: Int
    var finalBlockSizeK: Int?

    if precision.requiresQuantizationParameters {
      parameters = floatData.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
          fatalError(
            "Failed to obtain base address from floatData buffer for quantization parameter calculation."
          )
        }
        return precision.calculateQuantizationParameters(
          data: baseAddress,
          count: elementCount,
          shape: shape,
          mode: mode,
          strategy: strategy
        )
      }
      bufferSize = precision == .INT4 ? (elementCount + 1) / 2 : elementCount * precision.size

      // Extract blockSizeK from mode if it's blockwise
      if case let .blockwise(blockSize, _) = mode {
        finalBlockSizeK = blockSize
      }
    } else {
      // For non-quantized types (FP32, FP16, BF16), create dummy parameters
      parameters = QuantizationParameters(
        scale: 1.0,
        zeroPoint: 0,
        precision: precision,
        mode: mode,
        strategy: strategy
      )
      bufferSize = elementCount * precision.size
    }

    guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
      fatalError("Could not create quantized buffer")
    }

    floatData.withUnsafeBufferPointer { floatPtr in
      if precision.requiresQuantizationParameters {
        precision.quantize(
          input: floatPtr.baseAddress!,
          output: buffer.contents(),
          count: elementCount,
          parameters: parameters
        )
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

    return QuantizedTensor(
      device: device,
      data: buffer,
      parameters: parameters,
      elementCount: elementCount,
      shape: shape,
      blockScales: blockScales,
      blockZeroPoints: blockZeroPoints,
      blockSizeK: finalBlockSizeK,
      precomputedSums: precomputedSums
    )
  }

  /// Convert quantized tensor back to floating point
  /// - Returns: Array of floating point values
  public func toFloats() -> [Float] {
    var result = [Float](repeating: 0, count: elementCount)
    result.withUnsafeMutableBufferPointer { floatPtr in
      parameters.precision.dequantize(
        input: data.contents(),
        output: floatPtr.baseAddress!,
        count: elementCount,
        parameters: parameters
      )
    }
    return result
  }

  // MARK: - Codable Implementation

  /// Header structure for serialized QuantizedTensor
  private struct SerializationHeader: Codable {
    let version: UInt8
    let shape: [Int]
    let blockSizeK: Int?
    let quantMode: QuantizationMode
    let dtype: String
    let hasBlockScales: Bool
    let hasBlockZeroPoints: Bool
    let hasPrecomputedSums: Bool
    let elementCount: Int

    static let currentVersion: UInt8 = 1
  }

  private enum CodingKeys: String, CodingKey {
    case header
    case parameters
    case data
    case blockScales
    case blockZeroPoints
    case precomputedSums
  }

  /// Create aligned buffer with 64-byte alignment
  private static func createAlignedBuffer(device: MTLDevice, size: Int) -> MTLBuffer? {
    let alignedSize = (size + 63) & ~63 // Round up to 64-byte boundary
    return device.makeBuffer(length: alignedSize, options: .storageModeShared)
  }

  /// Serialize MTLBuffer contents to Data in little-endian format
  private func serializeBuffer(_ buffer: MTLBuffer) -> Data {
    let contents = buffer.contents()
    return Data(bytes: contents, count: buffer.length)
  }

  /// Deserialize Data to MTLBuffer with proper alignment
  private static func deserializeBuffer(from data: Data, device: MTLDevice) -> MTLBuffer? {
    guard let buffer = createAlignedBuffer(device: device, size: data.count) else {
      return nil
    }

    data.withUnsafeBytes { bytes in
      buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: data.count)
    }

    return buffer
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)

    // Create and encode header
    let header = SerializationHeader(
      version: SerializationHeader.currentVersion,
      shape: originalShape,
      blockSizeK: blockSizeK,
      quantMode: parameters.mode,
      dtype: String(describing: parameters.precision),
      hasBlockScales: blockScales != nil,
      hasBlockZeroPoints: blockZeroPoints != nil,
      hasPrecomputedSums: precomputedSums != nil,
      elementCount: elementCount
    )

    try container.encode(header, forKey: .header)
    try container.encode(parameters, forKey: .parameters)

    // Serialize main data buffer
    try container.encode(serializeBuffer(data), forKey: .data)

    // Serialize optional block-wise buffers
    if let blockScales {
      try container.encode(serializeBuffer(blockScales), forKey: .blockScales)
    }

    if let blockZeroPoints {
      try container.encode(serializeBuffer(blockZeroPoints), forKey: .blockZeroPoints)
    }

    if let precomputedSums {
      try container.encode(serializeBuffer(precomputedSums), forKey: .precomputedSums)
    }
  }

  public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    // Decode header
    let header = try container.decode(SerializationHeader.self, forKey: .header)

    // Validate version
    guard header.version == SerializationHeader.currentVersion else {
      throw DecodingError.dataCorrupted(
        DecodingError.Context(
          codingPath: decoder.codingPath,
          debugDescription: "Unsupported serialization version: \(header.version)"
        )
      )
    }

    // Decode parameters
    parameters = try container.decode(QuantizationParameters.self, forKey: .parameters)
    elementCount = header.elementCount
    originalShape = header.shape
    blockSizeK = header.blockSizeK

    // Retrieve the device that will own the decoded buffers.
    let device: MTLDevice
    if let box = decoder.userInfo[QuantizedTensorCoding.deviceKey] as? QuantizedTensorCoding.DeviceBox {
      device = box.device
    } else if let defaultDevice = MTLCreateSystemDefaultDevice() {
      device = defaultDevice
    } else {
      throw DecodingError.dataCorrupted(
        DecodingError.Context(
          codingPath: decoder.codingPath,
          debugDescription: "No Metal device available for buffer creation"
        )
      )
    }

    // Deserialize main data buffer
    let dataBytes = try container.decode(Data.self, forKey: .data)
    guard let dataBuffer = Self.deserializeBuffer(from: dataBytes, device: device) else {
      throw DecodingError.dataCorrupted(
        DecodingError.Context(
          codingPath: decoder.codingPath,
          debugDescription: "Failed to create data buffer"
        )
      )
    }
    data = dataBuffer

    // Deserialize optional block-wise buffers with 64-byte alignment
    if header.hasBlockScales {
      let scalesData = try container.decode(Data.self, forKey: .blockScales)
      blockScales = Self.deserializeBuffer(from: scalesData, device: device)
    } else {
      blockScales = nil
    }

    if header.hasBlockZeroPoints {
      let zeroPointsData = try container.decode(Data.self, forKey: .blockZeroPoints)
      blockZeroPoints = Self.deserializeBuffer(from: zeroPointsData, device: device)
    } else {
      blockZeroPoints = nil
    }

    if header.hasPrecomputedSums {
      let sumsData = try container.decode(Data.self, forKey: .precomputedSums)
      precomputedSums = Self.deserializeBuffer(from: sumsData, device: device)
    } else {
      precomputedSums = nil
    }
  }

  /// Convenience initializer for decoding with a specific device
  public static func decode(from data: Data, device: MTLDevice) throws -> QuantizedTensor {
    let decoder = JSONDecoder()

    // Store device in userInfo for access during decoding
    decoder.userInfo[QuantizedTensorCoding.deviceKey] = QuantizedTensorCoding.DeviceBox(device: device)

    return try decoder.decode(QuantizedTensor.self, from: data)
  }
}

import Metal

public enum SparseMQABuilder {
  public static func buildSlidingWindow(
    sequenceLength: Int,
    windowSize: Int,
    device: MTLDevice
  )
    -> MTLBuffer
  {
    let cappedWindow = max(1, windowSize)
    let length = sequenceLength * MemoryLayout<UInt32>.stride * 2
    guard let buffer = device.makeBuffer(length: length, options: [.storageModeShared]) else {
      fatalError("Unable to allocate sparse window buffer")
    }

    buffer.contents().withMemoryRebound(to: UInt32.self, capacity: sequenceLength * 2) { pointer in
      for index in 0..<sequenceLength {
        let halfWindow = cappedWindow / 2
        let start = max(0, index - halfWindow)
        let end = min(sequenceLength, index + halfWindow)
        pointer[index * 2] = UInt32(start)
        pointer[index * 2 + 1] = UInt32(end)
      }
    }

    return buffer
  }

  public static func buildBlockSparse(
    pattern: [[Bool]],
    blockSize: Int,
    device: MTLDevice
  )
    -> MTLBuffer
  {
    let rows = pattern.count
    let length = rows * MemoryLayout<UInt32>.stride * 2
    guard let buffer = device.makeBuffer(length: length, options: [.storageModeShared]) else {
      fatalError("Unable to allocate block-sparse window buffer")
    }

    buffer.contents().withMemoryRebound(to: UInt32.self, capacity: rows * 2) { pointer in
      for (rowIndex, rowPattern) in pattern.enumerated() {
        if
          let firstActive = rowPattern.firstIndex(of: true),
          let lastActive = rowPattern.lastIndex(of: true)
        {
          let start = UInt32(firstActive * blockSize)
          let maxColumns = rowPattern.count * blockSize
          let end = UInt32(min((lastActive + 1) * blockSize, maxColumns))
          pointer[rowIndex * 2] = start
          pointer[rowIndex * 2 + 1] = end
        } else {
          pointer[rowIndex * 2] = 0
          pointer[rowIndex * 2 + 1] = 0
        }
      }
    }

    return buffer
  }
}

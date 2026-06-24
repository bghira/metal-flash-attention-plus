//
//  MTLContext.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/26/24.
//

@preconcurrency import Metal

public struct MTLContext: @unchecked Sendable {
  public let device: MTLDevice
  public let commandQueue: MTLCommandQueue

  public static let global = MTLContext()

  public init() {
    device = MTLCreateSystemDefaultDevice()!
    commandQueue = device.makeCommandQueue()!
  }
}

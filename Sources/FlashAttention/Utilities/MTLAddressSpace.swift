//
//  MTLAddressSpace.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/9/24.
//

public enum MTLAddressSpace {
  case device
  case threadgroup

  public var keyword: String {
    switch self {
    case .device: "device"
    case .threadgroup: "threadgroup"
    }
  }

  public var offsetType: String {
    switch self {
    case .device: "uint"
    case .threadgroup: "ushort"
    }
  }
}

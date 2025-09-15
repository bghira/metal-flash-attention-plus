//
//  AttentionOperand.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/3/24.
//

/// The memory allocations used in attention kernels.
public enum AttentionOperand: Hashable, Equatable, CustomStringConvertible {
  case Q
  case K
  case S
  case P
  case V
  case O

  case L
  case D

  case dO
  case dV
  case dP
  case dS
  case dK
  case dQ

  /// The name in the shader source.
  ///
  /// Since the `AttentionOperand` type conforms to `CustomStringConvertible`,
  /// the name can be injected through string interpolation.
  public var description: String {
    switch self {
    case .Q: "Q"
    case .K: "K"
    case .S: "S"
    case .P: "P"
    case .V: "V"
    case .O: "O"
    case .L: "L"
    case .D: "D"
    case .dO: "dO"
    case .dV: "dV"
    case .dP: "dP"
    case .dS: "dS"
    case .dK: "dK"
    case .dQ: "dQ"
    }
  }

  public var bufferBinding: UInt8? {
    switch self {
    case .Q: 0
    case .K: 1
    case .S: nil
    case .P: nil
    case .V: 2
    case .O: 3
    case .L: 4
    case .D: 5
    case .dO: 6
    case .dV: 7
    case .dP: nil
    case .dS: nil
    case .dK: 8
    case .dQ: 9
    }
  }
}

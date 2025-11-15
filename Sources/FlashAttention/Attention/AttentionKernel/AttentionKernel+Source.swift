//
//  AttentionKernel+Source.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/2/24.
//

// Top level specification of the code structure.

import Foundation

public extension AttentionKernel {
  func createSource() -> String {
    func createLoop() -> String {
      switch type {
      case .forward:
        loopForward()
      case .backwardQuery:
        loopBackwardQuery()
      case .backwardKeyValue:
        loopBackwardKeyValue()
      case .mlaCompressed:
        "" // MLA uses a separate kernel, not this template
      }
    }

    let source = """

    \(createMetalSimdgroupEvent())
    \(createMetalSimdgroupMatrixStorage())
    using namespace metal;

    \(createConstants())

    // Declare the function.
    kernel void attention(
      \(createBufferBindings())
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],

      uint3 gid [[threadgroup_position_in_grid]],  // Now 3D: (block, head, batch)
      ushort sidx [[simdgroup_index_in_threadgroup]],
      ushort lane_id [[thread_index_in_simdgroup]]
    ) {
      ushort2 morton_offset = morton_order(lane_id);

      // Extract dimensions from 3D grid position
      uint block_id = gid.x;    // Sequence block index
      uint head_id = gid.y;     // Head index
      uint batch_id = gid.z;    // Batch index

      // For backward compatibility, compute original parallelization_group_offset
      uint parallelization_group_offset = block_id;
      parallelization_group_offset *= \(blockDimensions.parallelization);

      // Return early if the entire SIMD is out of bounds.
      if (\(parallelizationGroupOffset) >= \(parallelizationDimension)) {
        return;
      }

      // Check if multi-head parameters are provided
      // For single-head kernels created directly (not via MultiHeadAttention),
      // these pointers will be null
      if (num_heads_ptr != nullptr && num_kv_heads_ptr != nullptr &&
          head_dimension_ptr != nullptr && sequence_length_ptr != nullptr) {
        // Multi-head attention mode
        uint num_heads = *num_heads_ptr;
        uint num_kv_heads = *num_kv_heads_ptr;
        uint head_dimension = *head_dimension_ptr;
        uint sequence_length = *sequence_length_ptr;

        // Calculate buffer offsets for multi-head attention
        // Handle broadcast modes for K/V heads
        uint kv_head_id = head_id;
        if (num_kv_heads < num_heads) {
          // Grouped query attention or multi-query attention
          kv_head_id = head_id % num_kv_heads;
        }

        // Calculate offsets for this batch/head combination
        // Check if stride information is provided for non-contiguous tensor support
        uint q_batch_head_offset = 0;
        uint kv_batch_head_offset = 0;
        uint o_batch_head_offset = 0;

        if (Q_strides != nullptr) {
          // Use stride-based offset calculation for non-contiguous tensors
          // Assuming 4D tensor layout: [batch, seq, heads, dim] or [batch, heads, seq, dim]
          // Strides tell us how to calculate the actual memory offset
          q_batch_head_offset = batch_id * Q_strides[0] + head_id * Q_strides[2];
        } else {
          // Fallback to contiguous layout assumption
          q_batch_head_offset = (batch_id * num_heads + head_id) * sequence_length * head_dimension;
        }

        if (K_strides != nullptr && V_strides != nullptr) {
          kv_batch_head_offset = batch_id * K_strides[0] + kv_head_id * K_strides[2];
        } else {
          kv_batch_head_offset = (batch_id * num_kv_heads + kv_head_id) * sequence_length * head_dimension;
        }

        o_batch_head_offset = q_batch_head_offset;  // Output has same shape as query

        // Apply offsets to buffer pointers based on kernel type
        // Only apply offsets if we have multiple heads or batches
        if (num_heads > 1 || batch_id > 0) {
          \(createBufferOffsets())
        }
      }
      // Otherwise use single-head mode with original pointers

      \(createSetup())
      \(createLoop())
      \(createCleanup(type: type))
    }

    """

    // Force write source to file for debugging
    let patchedSource = GEMMBFloatHeaderEmbedder.embed(into: source)

    let sourceURL = URL(fileURLWithPath: "/tmp/quantized_attention_kernel.metal")
    do {
      try patchedSource.write(to: sourceURL, atomically: true, encoding: .utf8)
    } catch {}

    return patchedSource
  }
}

// MARK: - Function Signature

extension AttentionKernel {
  func createBufferOffsets() -> String {
    switch type {
    case .forward:
      """
      Q = Q + q_batch_head_offset;
      K = K + kv_batch_head_offset;
      V = V + kv_batch_head_offset;
      O = O + o_batch_head_offset;
      L = L + (batch_id * num_heads + head_id) * sequence_length;
      """
    case .backwardQuery:
      """
      Q = Q + q_batch_head_offset;
      K = K + kv_batch_head_offset;
      V = V + kv_batch_head_offset;
      O = O + o_batch_head_offset;
      dO = dO + o_batch_head_offset;
      dQ = dQ + q_batch_head_offset;
      L = L + (batch_id * num_heads + head_id) * sequence_length;
      D = D + (batch_id * num_heads + head_id) * sequence_length;
      """
    case .mlaCompressed:
      // MLA uses a separate kernel implementation
      ""
    case .backwardKeyValue:
      """
      Q = Q + q_batch_head_offset;
      K = K + kv_batch_head_offset;
      V = V + kv_batch_head_offset;
      dO = dO + o_batch_head_offset;
      dV = dV + kv_batch_head_offset;
      dK = dK + kv_batch_head_offset;
      L = L + (batch_id * num_heads + head_id) * sequence_length;
      D = D + (batch_id * num_heads + head_id) * sequence_length;
      """
    }
  }

  func createConstants() -> String {
    """

    // R = row dimension (output sequence)
    // C = column dimension (input sequence)
    constant uint R [[function_constant(0)]];
    constant uint C [[function_constant(1)]];

    // Sparsity pattern constants
    constant bool HAS_SLIDING_WINDOW [[function_constant(2)]];
    constant uint WINDOW_SIZE [[function_constant(3)]];
    constant bool IS_CAUSAL [[function_constant(4)]];

    // Blockwise quantization constants per operand
    constant bool HAS_BLOCKWISE_Q [[function_constant(5)]];
    constant bool HAS_BLOCKWISE_K [[function_constant(6)]];
    constant bool HAS_BLOCKWISE_V [[function_constant(7)]];
    constant uint BLOCK_SIZE_K [[function_constant(8)]];

    // Sparse masking and broadcast metadata
    constant bool HAS_SPARSE_RANGES [[function_constant(9)]];
    constant bool HAS_BLOCK_SPARSE [[function_constant(10)]];
    constant bool IS_MQA_MODE [[function_constant(11)]];
    constant uint NUM_KV_HEADS [[function_constant(12)]];

    """
  }

  func createBufferBindings() -> String {
    // What operands does the kernel use?
    var operands: [AttentionOperand] = []
    switch type {
    case .forward:
      // To simplify the implementation, we always compute log-sum-exp in the
      // forward pass. Even when it will never be used (model inference).
      // If this is an issue, clients can change the code to selectively
      // omit the 'L' operand.
      operands += [.Q, .K, .V, .O]
      operands += [.L]
    case .backwardQuery:
      operands += [.Q, .K, .V, .O]
      operands += [.dO, .dQ]
      operands += [.L, .D]
    case .backwardKeyValue:
      operands += [.Q, .K, .V]
      operands += [.dO, .dV, .dK]
      operands += [.L, .D]
    case .mlaCompressed:
      // MLA has different operands: Q, KV_latent, W_decompress_k, W_decompress_v, O
      // For compatibility, use standard operands
      operands += [.Q, .O]
    }
    operands.sort {
      $0.bufferBinding! < $1.bufferBinding!
    }

    var output = ""
    var currentBufferIndex = 0

    // First pass: regular operand buffers
    for operand in operands {
      let bufferIndex = operand.bufferBinding!
      currentBufferIndex = max(currentBufferIndex, Int(bufferIndex) + 1)

      var line = "device \(memoryName(operand))* \(operand) "
      line += "[[buffer(\(bufferIndex))]],"
      output += "  " + line + "\n"
    }

    // Second pass: quantization parameters for quantized operands
    for operand in operands where isQuantized(operand) {
      let operandName = "\(operand)".lowercased()

      // Scale parameter
      output += "  constant float &\(operandName)_scale [[buffer(\(currentBufferIndex))]], \n"
      currentBufferIndex += 1

      // Zero point parameter
      output +=
        "  constant int32_t &\(operandName)_zero_point [[buffer(\(currentBufferIndex))]], \n"
      currentBufferIndex += 1

      // Quantization strategy selector
      output +=
        "  constant uint &\(operandName)_strategy [[buffer(\(currentBufferIndex))]], \n"
      currentBufferIndex += 1

      // Quantization strategy version for forward compatibility
      output +=
        "  constant uint &\(operandName)_strategy_version [[buffer(\(currentBufferIndex))]], \n"
      currentBufferIndex += 1
    }

    // Third pass: blockwise quantization parameters for quantized operands
    for operand in operands where isQuantized(operand) {
      let operandName = "\(operand)".lowercased()

      // Block scales buffer (for per-block quantization)
      output +=
        "  device const float* \(operandName)_block_scales [[buffer(\(currentBufferIndex))]], \n"
      currentBufferIndex += 1

      // Block zero points buffer (for per-block quantization)
      output +=
        "  device const int32_t* \(operandName)_block_zero_points [[buffer(\(currentBufferIndex))]], \n"
      currentBufferIndex += 1
    }

    // Fourth pass: stride information for handling non-contiguous tensors
    // Add stride buffers for Q, K, V, O tensors to support PyTorch non-contiguous layouts
    let stridedOperands = [AttentionOperand.Q, AttentionOperand.K, AttentionOperand.V]
    for operand in stridedOperands {
      if !operands.contains(operand) {
        continue
      }

      output += "  constant int64_t* \(operand)_strides [[buffer(\(currentBufferIndex))]], \n"
      currentBufferIndex += 1
    }

    // Fifth pass: multi-head attention parameters (optional, with default values)
    output += "  constant uint *num_heads_ptr [[buffer(\(currentBufferIndex))]], \n"
    currentBufferIndex += 1
    output += "  constant uint *num_kv_heads_ptr [[buffer(\(currentBufferIndex))]], \n"
    currentBufferIndex += 1
    output += "  constant uint *head_dimension_ptr [[buffer(\(currentBufferIndex))]], \n"
    currentBufferIndex += 1
    output += "  constant uint *sequence_length_ptr [[buffer(\(currentBufferIndex))]], \n"
    currentBufferIndex += 1

    output += "  device char *mask_buffer_bytes [[buffer(\(currentBufferIndex))]], \n"
    currentBufferIndex += 1

    return output
  }
}

// MARK: - Outer Loop

// Forward
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//
//     O *= correction
//     load V[c]
//     O += P * V
//   }
//   O /= l
//
//   L = m + logBaseE(l)
//
// Backward Query
//   D = dO * O
//
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     P = exp(S - L)
//
//     load V[c]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//
//     load K[c]
//     dQ += dS * K
//   }
//
// Backward Key-Value
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//
//     load dO[r]
//     dV += P^T * dO
//
//     load dO[r]
//     load D[r]
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//
//     load Q[r]
//     dK += dS^T * Q
//   }

extension AttentionKernel {
  func loopForward() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .Q
    outerProductDesc.B = .K
    outerProductDesc.C = .S
    let QKT = outerProduct(descriptor: outerProductDesc)

    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .P
    accumulateDesc.B = .V
    accumulateDesc.C = .O
    accumulateDesc.everyIterationScale = "correction"
    accumulateDesc.lastIterationScale = "fast::divide(1, l)"
    let PV = accumulate(descriptor: accumulateDesc)

    return """

    // Outer loop over the traversal dimension.
    for (uint c = 0; c < C; c += \(blockDimensions.traversal)) {
      // S = Q * K^T
      \(QKT)
      \(maskAttentionMatrixEdge())
      \(applyExternalMask())
      \(maskSparsityPattern())

      // m = reduce(m)
      \(onlineReduceMaximum())

      // correction = exp(m_old) / exp(m_new)
      \(onlineCorrectO())

      // P = softmax(S * scaleFactor)
      \(optimizedSoftmax(derivative: false))

      // l = reduce(l)
      \(onlineReduceSum())

      // O *= correction
      // O += P * V
      // O /= l
      \(PV)
    }

    """
  }

  func loopBackwardQuery() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .Q
    outerProductDesc.B = .K
    outerProductDesc.C = .S
    let QKT = outerProduct(descriptor: outerProductDesc)

    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .dO
    outerProductDesc.B = .V
    outerProductDesc.C = .dP
    let dOVT = outerProduct(descriptor: outerProductDesc)

    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .dS
    accumulateDesc.B = .K
    accumulateDesc.C = .dQ
    let dSK = accumulate(descriptor: accumulateDesc)

    return """

    // Outer loop over the traversal dimension.
    for (uint c = 0; c < C; c += \(blockDimensions.traversal)) {
      // S = Q * K^T
      \(QKT)
      \(maskSparsityPattern())

      // P = softmax(S * scaleFactor)
      \(optimizedSoftmax(derivative: false))

      // dP = dO * V^T
      \(dOVT)

      // dS = P * (dP - D) * scaleFactor
      \(optimizedSoftmax(derivative: true))

      // dQ += dS * K
      \(dSK)
    }

    """
  }

  func loopBackwardKeyValue() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .K
    outerProductDesc.B = .Q
    outerProductDesc.C = .S // S^T
    let KQT = outerProduct(descriptor: outerProductDesc)

    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .P // P^T
    accumulateDesc.B = .dO
    accumulateDesc.C = .dV
    let PTdO = accumulate(descriptor: accumulateDesc)

    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .V
    outerProductDesc.B = .dO
    outerProductDesc.C = .dP // dP^T
    let VdOT = outerProduct(descriptor: outerProductDesc)

    accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .dS // dS^T
    accumulateDesc.B = .Q
    accumulateDesc.C = .dK
    let dSTQ = accumulate(descriptor: accumulateDesc)

    return """

    // Outer loop over the traversal dimension.
    for (uint r = 0; r < R; r += \(blockDimensions.traversal)) {
      // S^T = K * Q^T
      \(KQT)
      \(maskSparsityPatternTransposed())

      // P^T = exp(S^T - L)
      \(optimizedSoftmax(derivative: false))

      // dV += P^T * dO
      \(PTdO)

      // dP^T = V * dO^T
      \(VdOT)

      // dS^T = P^T * (dP^T - D) * scaleFactor
      \(optimizedSoftmax(derivative: true))

      // dK += dS^T * Q
      \(dSTQ)
    }

    """
  }
}

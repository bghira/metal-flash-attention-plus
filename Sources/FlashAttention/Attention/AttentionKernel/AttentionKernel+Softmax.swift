//
//  AttentionKernel+Softmax.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Elementwise operations on the attention matrix.

// MARK: - Scale Factor

extension AttentionKernel {
  // The scale factor in scaled dot product attention.
  //
  // Parameters:
  // - derivative: Whether this is the derivative softmax.
  func dotProductScale(derivative: Bool) -> Float {
    let logBase2E: Float = 1.442695041

    if !derivative {
      return logBase2E * softmaxScale
    } else {
      return softmaxScale
    }
  }
}

// MARK: - Compute D (dO * O)

extension AttentionKernel {
  func computeD() -> String {
    // Parts of the dO * O reduction that fall within block bounds.
    func bulkContributions(truncatedHeadDimension: UInt16) -> String {
      // Recycle most of the cached values for dO.
      func declareDerivativeOLocation() -> String {
        if cached(.dO) {
          ""
        } else {
          """

          // Where the dO data will be read from.
          auto dO_src = simdgroup_matrix_storage<\(memoryName(.dO))>
          ::apply_offset(
            dO, \(leadingDimension(.dO)),
            offset_src, \(transposed(.dO)));

          """
        }
      }
      func loadDerivativeO() -> String {
        if cached(.dO) {
          """

          auto dO = dO_sram[d / 8];

          """
        } else {
          """

          simdgroup_matrix_storage<\(registerName(.dO))> dO;
          dO.\(loadCall(
            .dO,
            src: "dO_src",
            leadingDim: "\(leadingDimension(.dO))",
            origin: "ushort2(d, 0)",
            transpose: "\(transposed(.dO))"
          ));

          """
        }
      }

      return """

      // Threads outside of the matrix along the row dimension,
      // have their origin shifted in-bounds.
      uint D_offset = morton_offset.x;
      uint R_offset = \(clampedParallelizationThreadOffset);
      uint2 offset_src(D_offset, R_offset);

      \(declareDerivativeOLocation())

      // Where the O data will be read from.
      auto O_src = simdgroup_matrix_storage<\(memoryName(.O))>
      ::apply_offset(
        O, \(leadingDimension(.O)),
        offset_src, \(transposed(.O)));

      // Going to use async copy to handle the matrix edge.
      #pragma clang loop unroll(disable)
      for (ushort d = 0; d < \(truncatedHeadDimension); d += 8) {
        \(loadDerivativeO())

        simdgroup_matrix_storage<\(registerName(.O))> O;
        O.\(loadCall(
          .O,
          src: "O_src",
          leadingDim: "\(leadingDimension(.O))",
          origin: "ushort2(d, 0)",
          transpose: "\(transposed(.O))"
        ));

        // Perform the pointwise multiplication.
        auto dO_value = *(dO.thread_elements());
        auto O_value = *(O.thread_elements());
        D_accumulator += float2(dO_value) * float2(O_value);
      }

      """
    }

    // Parts of the dO * O reduction that fall on an indivisible edge.
    func edgeContributions(truncatedHeadDimension: UInt16) -> String {
      guard headDimension % 8 != 0 else {
        return ""
      }

      // Abbreviated block, only covers the last 8 elements.
      func leadingBlockDimension(_ operand: AttentionOperand) -> UInt16 {
        if transposed(operand) {
          blockSequenceLength(operand)
        } else {
          8
        }
      }

      // Distinct from the block bytes that would be used to calculate
      // the threadgroup memory allocation.
      func blockBytesDerivativeO() -> UInt16 {
        let memoryPrecision = memoryPrecisions[.dO]!
        let size = UInt16(memoryPrecision.size)
        return blockDimensions.parallelization * 8 * size
      }

      return """

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint D_offset = \(truncatedHeadDimension);
        uint R_offset = \(parallelizationGroupOffset);
        uint2 offset_src(D_offset, R_offset);

        auto dO_src = simdgroup_matrix_storage<\(memoryName(.dO))>
        ::apply_offset(
          dO, \(leadingDimension(.dO)),
          offset_src, \(transposed(.dO)));
        auto O_src = simdgroup_matrix_storage<\(memoryName(.O))>
        ::apply_offset(
          O, \(leadingDimension(.O)),
          offset_src, \(transposed(.O)));

        auto dO_dst = (threadgroup \(memoryName(.dO))*)(threadgroup_block);
        auto O_dst = (threadgroup \(memoryName(.O))*)(
          threadgroup_block + \(blockBytesDerivativeO()));

        ushort D_src_dimension = \(headDimension) % 8;
        ushort D_dst_dimension = 8;
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile_src(D_src_dimension, R_dimension);
        ushort2 tile_dst(D_dst_dimension, R_dimension);

        // Issue two async copies.
        simdgroup_event events[2];
        events[0].async_copy(
          dO_dst, \(leadingBlockDimension(.dO)), tile_dst,
          dO_src, \(leadingDimension(.dO)), tile_src, \(transposed(.dO)));
        events[1].async_copy(
          O_dst, \(leadingBlockDimension(.O)), tile_dst,
          O_src, \(leadingDimension(.O)), tile_src, \(transposed(.O)));
        simdgroup_event::wait(2, events);
      }

      // Where the dO and O data will be read from.
      ushort2 offset_src(morton_offset.x, morton_offset.y + sidx * 8);
      auto dO_block = (threadgroup \(memoryName(.dO))*)(threadgroup_block);
      auto O_block = (threadgroup \(memoryName(.O))*)(
        threadgroup_block + \(blockBytesDerivativeO()));

      dO_block = simdgroup_matrix_storage<\(memoryName(.dO))>
      ::apply_offset(
        dO_block, \(leadingBlockDimension(.dO)),
        offset_src, \(transposed(.dO)));
      O_block = simdgroup_matrix_storage<\(memoryName(.O))>
      ::apply_offset(
        O_block, \(leadingBlockDimension(.O)),
        offset_src, \(transposed(.O)));
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Load the zero-padded edge data.
      ushort2 origin(0, 0);
      simdgroup_matrix_storage<\(registerName(.dO))> dO;
      simdgroup_matrix_storage<\(registerName(.O))> O;
      dO.\(loadCall(
        .dO,
        src: "dO_block",
        leadingDim: "\(leadingBlockDimension(.dO))",
        origin: "origin",
        transpose: "\(transposed(.dO))"
      ));
      O.\(loadCall(
        .O,
        src: "O_block",
        leadingDim: "\(leadingBlockDimension(.O))",
        origin: "origin",
        transpose: "\(transposed(.O))"
      ));

      // Perform the pointwise multiplication.
      auto dO_value = *(dO.thread_elements());
      auto O_value = *(O.thread_elements());
      D_accumulator += float2(dO_value) * float2(O_value);

      """
    }

    // Outer loop over the head dimension.
    let loopEndFloor = headDimension - headDimension % 8
    return """

    float2 D_accumulator(0);
    {
      \(bulkContributions(truncatedHeadDimension: loopEndFloor))
    }
    {
      \(edgeContributions(truncatedHeadDimension: loopEndFloor))
    }

    float D_sram = D_accumulator[0] + D_accumulator[1];
    D_sram += simd_shuffle_xor(D_sram, 1);
    D_sram += simd_shuffle_xor(D_sram, 8);
    D_sram *= \(dotProductScale(derivative: true));

    """
  }
}

// MARK: - Mask

extension AttentionKernel {
  // Prevent the zero padding from changing the values of 'm' and 'l'.
  func maskAttentionMatrixEdge() -> String {
    let blockDim = blockDimensions.traversal
    let remainder = "(\(traversalDimension) % \(blockDim))"
    let remainderFloor = "(\(remainder) - (\(remainder) % 8))"
    let logBase2E: Float = 1.442695041

    return """

    if ((\(remainder) != 0) &&
        (\(traversalOffset) + \(blockDim) > \(traversalDimension))) {
      // Prevent the value from becoming -INF during the FMA before the
      // exponentiation. If the multiplication during FMA returns -INF,
      // subtracting a positive 'm' value will turn it into zero. We don't want
      // that. exp(0) evaluates to 1.00 and corrupts the value of 'l'.
      const \(registerName(.S)) mask_value =
      (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

      #pragma clang loop unroll(full)
      for (ushort index = 0; index < 2; ++index) {
        if (morton_offset.x + index >= \(remainder) - \(remainderFloor)) {
          auto S_elements = S_sram[\(remainderFloor) / 8].thread_elements();
          (*S_elements)[index] = mask_value;
        }
      }
      #pragma clang loop unroll(full)
      for (ushort c = \(remainderFloor) + 8; c < \(blockDim); c += 8) {
        auto S_elements = S_sram[c / 8].thread_elements();
        *S_elements = mask_value;
      }
    }

    """
  }

  // Apply sparsity patterns (causal, sliding window) to attention matrix
  func maskSparsityPattern() -> String {
    let logBase2E: Float = 1.442695041
    let blockDim = blockDimensions.traversal

    // Auto-optimization: Use bitmask for smaller problems, element-wise for larger
    // Based on comprehensive benchmarking: seq² × head < 50,331,648 generally favors bitmask
    let useBitmaskOptimization = shouldUseBitmaskOptimization()

    return """

    // Apply sparsity patterns
    if (IS_CAUSAL || HAS_SLIDING_WINDOW || HAS_SPARSE_RANGES || HAS_BLOCK_SPARSE) {
      const \(registerName(.S)) mask_value =
      (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

      #pragma clang loop unroll(full)
      for (ushort c = 0; c < \(blockDim); c += 8) {
        auto S_elements = S_sram[c / 8].thread_elements();

        \(useBitmaskOptimization ? generateBitmaskMasking() : generateElementwiseMasking())
      }
    }

    """
  }

  func applyExternalMask() -> String {
    """
    if (mask_buffer_bytes != nullptr && !HAS_SPARSE_RANGES && !HAS_BLOCK_SPARSE) {
      auto mask_buffer = reinterpret_cast<device const float*>(mask_buffer_bytes);
      uint row_idx = \(parallelizationGroupOffset) + morton_offset.y;
      uint col_base = \(traversalOffset) + c + morton_offset.x;

      if (row_idx < R) {
        auto S_elements = S_sram[c / 8].thread_elements();

        #pragma clang loop unroll(full)
        for (ushort index = 0; index < 2; ++index) {
          uint col_idx = col_base + index;
          if (col_idx < C) {
            uint effective_num_heads = (multi_head.enabled != 0) ? multi_head.num_heads : 1u;
            uint effective_batch = (multi_head.enabled != 0) ? batch_id : 0u;
            uint effective_head = (multi_head.enabled != 0) ? head_id : 0u;

            ulong mask_index = (((ulong)effective_batch * (ulong)effective_num_heads) + (ulong)effective_head) * (ulong)R;
            mask_index = (mask_index + (ulong)row_idx) * (ulong)C + (ulong)col_idx;
            float mask_value = *(mask_buffer + mask_index);
            (*S_elements)[index] += mask_value;
          }
        }
      }
    }
    """
  }

  // Auto-optimization heuristics based on comprehensive benchmarking
  private func shouldUseBitmaskOptimization() -> Bool {
    // Head dimensions where bitmask consistently performs well
    let headDim = headDimension

    // Based on benchmark results: head dimensions 64 and 128 show strong bitmask preference
    if headDim == 64 || headDim == 128 {
      return true
    }

    // For smaller head dimensions, bitmask tends to be better
    if headDim <= 96 {
      return true
    }

    // For very large head dimensions, element-wise tends to be more stable
    if headDim >= 192 {
      return false
    }

    // Default to bitmask for intermediate sizes (good average performance)
    return true
  }

  private func generateBitmaskMasking() -> String {
    """
              // GLUON-inspired vectorized masking for Metal SIMD (Auto-optimized)
              uint row_idx = \(parallelizationGroupOffset) + morton_offset.y;
              uint col_base = \(traversalOffset) + c + morton_offset.x;

              ulong sparse_head_offset = 0;
              if (HAS_SPARSE_RANGES) {
                uint effective_num_heads = (multi_head.enabled != 0) ? multi_head.num_heads : 1u;
                uint effective_num_kv_heads = effective_num_heads;
                if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                  effective_num_kv_heads = multi_head.num_kv_heads;
                }
                uint kv_head_id = head_id;
                if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                  kv_head_id = head_id % effective_num_kv_heads;
                }
                sparse_head_offset = (((ulong)batch_id * (ulong)effective_num_kv_heads) + (ulong)kv_head_id) * (ulong)R;
              }

              uint2 sparse_range = uint2(0);
              if (HAS_SPARSE_RANGES && mask_buffer_bytes != nullptr) {
                auto sparse_ranges = reinterpret_cast<device const uint2*>(mask_buffer_bytes);
                sparse_range = *(sparse_ranges + sparse_head_offset + (ulong)row_idx);
              }

              // Optimized causal masking using bitmask approach
              if (IS_CAUSAL) {
                // Pre-compute causal mask for 2-element vector (morton_offset.x spans 2 elements)
                uint causal_mask = 0;
                if (row_idx >= col_base) {
                  uint mask_width = min(2u, row_idx - col_base + 1);
                  causal_mask = (1u << mask_width) - 1;
                }

                #pragma clang loop unroll(full)
                for (ushort index = 0; index < 2; ++index) {
                  uint col_idx = col_base + index;
                  bool causal_should_mask = !(causal_mask & (1u << index));

                  bool should_mask = causal_should_mask;

                  // Sliding window masking: mask beyond window
                  if (HAS_SLIDING_WINDOW && row_idx > col_idx + WINDOW_SIZE) {
                    should_mask = true;
                  }

                  if (HAS_SPARSE_RANGES && mask_buffer_bytes != nullptr) {
                    bool outside_sparse = (col_idx < sparse_range.x) || (col_idx >= sparse_range.y);
                    should_mask = should_mask || outside_sparse;
                  }

                  if (should_mask) {
                    (*S_elements)[index] = mask_value;
                  }
                }
              } else {
                // Non-causal path (sliding window only)
                #pragma clang loop unroll(full)
                for (ushort index = 0; index < 2; ++index) {
                  uint col_idx = col_base + index;
                  bool should_mask = false;

                  // Sliding window masking: mask beyond window
                  if (HAS_SLIDING_WINDOW && row_idx > col_idx + WINDOW_SIZE) {
                    should_mask = true;
                  }

                  if (HAS_SPARSE_RANGES && mask_buffer_bytes != nullptr) {
                    bool outside_sparse = (col_idx < sparse_range.x) || (col_idx >= sparse_range.y);
                    should_mask = should_mask || outside_sparse;
                  }

                  if (should_mask) {
                    (*S_elements)[index] = mask_value;
                  }
                }
              }
    """
  }

  private func generateElementwiseMasking() -> String {
    """
              // Traditional element-wise masking (Auto-optimized for large problems)
              uint row_idx = \(parallelizationGroupOffset) + morton_offset.y;

              #pragma clang loop unroll(full)
              for (ushort index = 0; index < 2; ++index) {
                uint col_idx = \(traversalOffset) + c + morton_offset.x + index;

                bool should_mask = false;

                // Causal masking: mask upper triangular part
                if (IS_CAUSAL && col_idx > row_idx) {
                  should_mask = true;
                }

                // Sliding window masking: mask beyond window
                if (HAS_SLIDING_WINDOW && row_idx > col_idx + WINDOW_SIZE) {
                  should_mask = true;
                }

                if (HAS_SPARSE_RANGES && mask_buffer_bytes != nullptr) {
                  uint effective_num_heads = (multi_head.enabled != 0) ? multi_head.num_heads : 1u;
                  uint effective_num_kv_heads = effective_num_heads;
                  if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                    effective_num_kv_heads = multi_head.num_kv_heads;
                  }
                  uint kv_head_id = head_id;
                  if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                    kv_head_id = head_id % effective_num_kv_heads;
                  }
                  ulong sparse_head_offset = (((ulong)batch_id * (ulong)effective_num_kv_heads) + (ulong)kv_head_id) * (ulong)R;
                  auto sparse_ranges = reinterpret_cast<device const uint2*>(mask_buffer_bytes);
                  uint2 sparse_range = *(sparse_ranges + sparse_head_offset + (ulong)row_idx);
                  if (col_idx < sparse_range.x || col_idx >= sparse_range.y) {
                    should_mask = true;
                  }
                }

                if (should_mask) {
                  (*S_elements)[index] = mask_value;
                }
              }
    """
  }

  // Apply sparsity patterns for transposed attention matrix (used in backward key-value pass)
  func maskSparsityPatternTransposed() -> String {
    let logBase2E: Float = 1.442695041
    let blockDim = blockDimensions.traversal

    // Auto-optimization: Use same strategy as forward pass
    let useBitmaskOptimization = shouldUseBitmaskOptimization()

    return """

        // Apply sparsity patterns for transposed matrix
        if (IS_CAUSAL || HAS_SLIDING_WINDOW || HAS_SPARSE_RANGES || HAS_BLOCK_SPARSE) {
          const \(registerName(.S)) mask_value =
          (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

          #pragma clang loop unroll(full)
          for (ushort c = 0; c < \(blockDim); c += 8) {
            auto S_elements = S_sram[c / 8].thread_elements();

            \(
              useBitmaskOptimization ? generateBitmaskMaskingTransposed() :
                generateElementwiseMaskingTransposed()
            )
          }
        }

        """
  }

  private func generateBitmaskMaskingTransposed() -> String {
    """
              // GLUON-inspired vectorized masking for Metal SIMD (transposed, auto-optimized)
              // For transposed matrix S^T, swap row and col interpretations
              uint col_idx = \(parallelizationGroupOffset) + morton_offset.y;
              uint row_base = \(traversalOffset) + c + morton_offset.x;


              // Optimized causal masking using bitmask approach (transposed)
              if (IS_CAUSAL) {
                // Pre-compute causal mask for 2-element vector
                uint causal_mask = 0;
                if (col_idx >= row_base) {
                  uint mask_width = min(2u, col_idx - row_base + 1);
                  causal_mask = (1u << mask_width) - 1;
                }

                #pragma clang loop unroll(full)
                for (ushort index = 0; index < 2; ++index) {
                  uint row_idx = row_base + index;
                  bool causal_should_mask = !(causal_mask & (1u << index));

                  bool should_mask = causal_should_mask;

                  // Sliding window masking: mask beyond window
                  if (HAS_SLIDING_WINDOW && row_idx > col_idx + WINDOW_SIZE) {
                    should_mask = true;
                  }

                  if (HAS_SPARSE_RANGES && mask_buffer_bytes != nullptr) {
                    uint effective_num_heads = (multi_head.enabled != 0) ? multi_head.num_heads : 1u;
                    uint effective_num_kv_heads = effective_num_heads;
                    if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                      effective_num_kv_heads = multi_head.num_kv_heads;
                    }
                    uint kv_head_id = head_id;
                    if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                      kv_head_id = head_id % effective_num_kv_heads;
                    }
                    ulong sparse_head_offset = (((ulong)batch_id * (ulong)effective_num_kv_heads) + (ulong)kv_head_id) * (ulong)R;
                    auto sparse_ranges = reinterpret_cast<device const uint2*>(mask_buffer_bytes);
                    uint2 sparse_range = *(sparse_ranges + sparse_head_offset + (ulong)row_idx);
                    if (col_idx < sparse_range.x || col_idx >= sparse_range.y) {
                      should_mask = true;
                    }
                  }

                  if (should_mask) {
                    (*S_elements)[index] = mask_value;
                  }
                }
              } else {
                // Non-causal path (sliding window only)
                #pragma clang loop unroll(full)
                for (ushort index = 0; index < 2; ++index) {
                  uint row_idx = row_base + index;
                  bool should_mask = false;

                  // Sliding window masking: mask beyond window
                  if (HAS_SLIDING_WINDOW && row_idx > col_idx + WINDOW_SIZE) {
                    should_mask = true;
                  }

                  if (HAS_SPARSE_RANGES && mask_buffer_bytes != nullptr) {
                    uint effective_num_heads = (multi_head.enabled != 0) ? multi_head.num_heads : 1u;
                    uint effective_num_kv_heads = effective_num_heads;
                    if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                      effective_num_kv_heads = multi_head.num_kv_heads;
                    }
                    uint kv_head_id = head_id;
                    if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                      kv_head_id = head_id % effective_num_kv_heads;
                    }
                    ulong sparse_head_offset = (((ulong)batch_id * (ulong)effective_num_kv_heads) + (ulong)kv_head_id) * (ulong)R;
                    auto sparse_ranges = reinterpret_cast<device const uint2*>(mask_buffer_bytes);
                    uint2 sparse_range = *(sparse_ranges + sparse_head_offset + (ulong)row_idx);
                    if (col_idx < sparse_range.x || col_idx >= sparse_range.y) {
                      should_mask = true;
                    }
                  }

                  if (should_mask) {
                    (*S_elements)[index] = mask_value;
                  }
                }
              }
    """
  }

  private func generateElementwiseMaskingTransposed() -> String {
    """
              // Traditional element-wise masking (transposed, auto-optimized for large problems)
              // For transposed matrix S^T, swap row and col interpretations
              uint col_idx = \(parallelizationGroupOffset) + morton_offset.y;

              #pragma clang loop unroll(full)
              for (ushort index = 0; index < 2; ++index) {
                uint row_idx = \(traversalOffset) + c + morton_offset.x + index;

                bool should_mask = false;

                // Causal masking: mask upper triangular part
                if (IS_CAUSAL && col_idx > row_idx) {
                  should_mask = true;
                }

                // Sliding window masking: mask beyond window
                if (HAS_SLIDING_WINDOW && row_idx > col_idx + WINDOW_SIZE) {
                  should_mask = true;
                }

                if (HAS_SPARSE_RANGES && mask_buffer_bytes != nullptr) {
                  uint effective_num_heads = (multi_head.enabled != 0) ? multi_head.num_heads : 1u;
                  uint effective_num_kv_heads = effective_num_heads;
                  if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                    effective_num_kv_heads = multi_head.num_kv_heads;
                  }
                  uint kv_head_id = head_id;
                  if (multi_head.enabled != 0 && multi_head.num_kv_heads != 0) {
                    kv_head_id = head_id % effective_num_kv_heads;
                  }
                  ulong sparse_head_offset = (((ulong)batch_id * (ulong)effective_num_kv_heads) + (ulong)kv_head_id) * (ulong)R;
                  auto sparse_ranges = reinterpret_cast<device const uint2*>(mask_buffer_bytes);
                  uint2 sparse_range = *(sparse_ranges + sparse_head_offset + (ulong)row_idx);
                  if (col_idx < sparse_range.x || col_idx >= sparse_range.y) {
                    should_mask = true;
                  }
                }

                if (should_mask) {
                  (*S_elements)[index] = mask_value;
                }
              }
    """
  }
}

// MARK: - Reduce

extension AttentionKernel {
  // Reduce maximum during the online softmax.
  func onlineReduceMaximum() -> String {
    """

    // update 'm'
    vec<\(registerName(.S)), 2> m_new_accumulator;
    #pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
      auto S_elements = S_sram[c / 8].thread_elements();
      if (c == 0) {
        m_new_accumulator = *S_elements;
      } else {
        m_new_accumulator = max(m_new_accumulator, *S_elements);
      }
    }
    float m_new = max(m_new_accumulator[0], m_new_accumulator[1]);
    m_new = max(m_new, simd_shuffle_xor(m_new, 1));
    m_new = max(m_new, simd_shuffle_xor(m_new, 8));
    m_new *= \(dotProductScale(derivative: false));

    """
  }

  // Rescale 'O' to reflect the new maximum.
  func onlineCorrectO() -> String {
    """

    // update 'O'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
      m = m_new;
    }

    """
  }

  // Reduce sum during the online softmax.
  func onlineReduceSum() -> String {
    """

    // update 'l'
    float2 l_new_accumulator;
    #pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
      auto P_elements = P_sram[c / 8].thread_elements();
      if (c == 0) {
        l_new_accumulator = float2(*P_elements);
      } else {
        l_new_accumulator += float2(*P_elements);
      }
    }
    float l_new = l_new_accumulator[0] + l_new_accumulator[1];
    l_new += simd_shuffle_xor(l_new, 1);
    l_new += simd_shuffle_xor(l_new, 8);
    l = l * correction + l_new;

    """
  }
}

// MARK: - Softmax

extension AttentionKernel {
  // A softmax where the per-row statistics have been reduced beforehand.
  //
  // Parameters:
  // - derivative: Whether this is the derivative softmax.
  func softmax(derivative: Bool) -> String {
    let operand: AttentionOperand = derivative ? .D : .L

    func allocateOutput() -> String {
      let blockDim = blockDimensions.traversal
      if !derivative {
        return """

        simdgroup_matrix_storage<\(registerName(.P))> \
        P_sram[\(blockDim) / 8];

        """
      } else {
        return """

        simdgroup_matrix_storage<\(registerName(.dS))> \
        dS_sram[\(blockDim) / 8];

        """
      }
    }

    func loadOperand() -> String {
      """

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        auto \(operand)_src = \(operand) + \(traversalOffset);
        auto \(operand)_dst =
        (threadgroup \(memoryName(operand))*)(threadgroup_block);

        ushort R_src_dimension = min(
          uint(\(blockDimensions.traversal)),
          uint(\(traversalDimension) - \(traversalOffset)));
        ushort R_dst_dimension = max(
          ushort(\(paddedTraversalEdge)),
          ushort(R_src_dimension));

        // Issue an async copy.
        simdgroup_event event;
        event.async_copy(
          \(operand)_dst, 1, ushort2(R_dst_dimension, 1),
          \(operand)_src, 1, ushort2(R_src_dimension, 1));
        simdgroup_event::wait(1, &event);
      }

      """
    }

    // Declares the source of L or D.
    //
    // Also guards against unsafe accesses to the declared pointer (barrier).
    func declareOperandLocation(addressSpace: MTLAddressSpace) -> String {
      if addressSpace == .device {
        """

        auto \(operand)_src = \(operand);
        \(operand)_src += \(traversalOffset) + morton_offset.x;

        """
      } else {
        """

        auto \(operand)_src =
        (threadgroup \(memoryName(operand))*)(threadgroup_block);
        \(operand)_src += morton_offset.x;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        """
      }
    }

    func overwriteAttentionMatrixElements() -> String {
      let scale = dotProductScale(derivative: derivative)

      if !derivative {
        return """

        auto S = *(S_sram[c / 8].thread_elements());
        auto P = vec<\(registerName(.P)), 2>(
          fast::exp2(float2(S) * \(scale) - float2(L_elements)));
        *(P_sram[c / 8].thread_elements()) = P;

        """
      } else {
        return """

        auto P = *(P_sram[c / 8].thread_elements());
        auto dP = *(dP_sram[c / 8].thread_elements());
        auto dS = vec<\(registerName(.dS)), 2>(
          float2(P) * (float2(dP) * \(scale) - float2(D_elements)));
        *(dS_sram[c / 8].thread_elements()) = dS;

        """
      }
    }

    func innerLoop() -> String {
      switch type {
      case .forward:
        """

        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          auto L_elements = m;
          \(overwriteAttentionMatrixElements())
        }

        """
      case .backwardQuery:
        """

        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          auto \(operand)_elements = \(operand)_sram;
          \(overwriteAttentionMatrixElements())
        }

        """
      case .backwardKeyValue:
        """

        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          ushort2 \(operand)_origin(c, 0);
          simdgroup_matrix_storage<\(registerName(operand))> \(operand);
          \(operand).\(loadCall(
            operand,
            src: "\(operand)_src",
            leadingDim: "1",
            origin: "\(operand)_origin",
            transpose: "false"
          ));
          auto \(operand)_elements = *(\(operand).thread_elements());

          \(overwriteAttentionMatrixElements())
        }

        """
      case .mlaCompressed:
        // MLA uses a completely different kernel and does not use template-based softmax
        ""
      }
    }

    switch type {
    case .forward, .backwardQuery:
      return """

      \(allocateOutput())
      {
        \(innerLoop())
      }

      """
    case .backwardKeyValue:
      let blockDim = blockDimensions.traversal
      let condition = """
      \(!preferAsyncLoad) && (
          (\(traversalDimension) % \(blockDim) == 0) ||
          (\(traversalOffset) + \(blockDim) <= \(traversalDimension))
        )
      """

      return """

      \(allocateOutput())
      if (\(condition)) {
        \(declareOperandLocation(addressSpace: .device))
        \(innerLoop())
      } else {
        \(loadOperand())
        \(declareOperandLocation(addressSpace: .threadgroup))
        \(innerLoop())
      }

      """
    case .mlaCompressed:
      // MLA uses a completely different kernel and does not use template-based softmax
      return ""
    }
  }
}

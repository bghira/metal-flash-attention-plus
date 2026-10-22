//
//  AttentionKernel+FastForward.swift
//  FlashAttention
//
//  Fast forward attention loop, tuned on Apple M5 Pro:
//
//  1. Q fragments are loaded from device memory once, before the traversal
//     loop, and stay register-resident (gated to head dimensions up to 128).
//     This eliminates the per-tile Q staging and fragment loads.
//  2. K/V tiles are staged through double-buffered threadgroup memory.
//     While iteration t computes, the tile for iteration t+1 is fetched
//     into registers with batched 16-byte vector loads, hiding global
//     memory latency behind tensor-core work.
//  3. K lands transposed ([head x seq]) in threadgroup memory, so the
//     B-operand fragment load is one packed vector load instead of two
//     scalar loads 'head' elements apart. The row stride carries a
//     two-element pad to keep the staging scatter out of a single
//     threadgroup bank group; V stays row-major.
//  4. Staging uses two phases (read all global data into registers, then
//     write all data to threadgroup memory), avoiding serialized
//     load-use stalls.
//  5. One threadgroup barrier per traversal iteration.
//  6. Causal attention stops at the diagonal tile instead of traversing
//     the fully masked remainder of the sequence.
//
//  Gated to non-quantized forward kernels whose head dimension is a
//  multiple of 8 (up to 128) and whose double-buffered K/V tiles fit in
//  threadgroup memory. Masking is handled by the existing mask
//  generators. Strided views whose row stride is not a multiple of 8
//  elements fall back to scalar staging at runtime.
//
//  Note: the vectorized staging path assumes 16-byte-aligned K/V buffer
//  bases, which holds for all realistic allocations (Metal guarantees
//  at least 4 bytes; PyTorch MPS and MFA's own allocations are 16+).

import Metal

extension AttentionKernel {
  var useFastForwardPath: Bool {
    type == .forward && preferFastForward
  }

  var fastForwardThreads: Int {
    32 * Int(blockDimensions.parallelization / 8)
  }

  // K is staged transposed ([head][seq]) so the B-operand fragment load
  // reads two adjacent sequence elements with one vector load instead of
  // two scalar loads `head` elements apart. The two-element padding keeps
  // the transposed staging stores out of the same threadgroup bank (an
  // unpadded row stride is a multiple of the 128-byte bank period, which
  // serializes the scatter eight ways).
  var fastForwardKStride: Int {
    Int(blockDimensions.traversal) + 2
  }

  // Double-buffered K ([head][stride]) plus double-buffered V ([seq][head]).
  var fastForwardThreadgroupBytes: Int {
    let head = Int(paddedHeadDimension)
    let traversal = Int(blockDimensions.traversal)
    let elemSize = Int(memoryPrecisions[.K]!.size)
    return 2 * (head * fastForwardKStride + traversal * head) * elemSize
  }

  func loopForwardFast() -> String {
    let traversal = Int(blockDimensions.traversal)
    let head = Int(paddedHeadDimension)
    let strideK = fastForwardKStride
    let threads = fastForwardThreads
    let elemSize = Int(memoryPrecisions[.K]!.size)
    let elemsPerChunk = 16 / elemSize
    let chunksPerRow = head / elemsPerChunk
    let chunksPerTile = traversal * chunksPerRow
    let chunksPerThread = (chunksPerTile + threads - 1) / threads
    let kTileElems = head * strideK
    let vTileElems = traversal * head
    let kBytes = 2 * kTileElems * elemSize
    let memK = memoryName(.K)
    let memV = memoryName(.V)
    let memQ = memoryName(.Q)
    let memO = memoryName(.O)
    let regQ = registerName(.Q)
    let regK = registerName(.K)
    let regV = registerName(.V)
    let regS = registerName(.S)
    let regP = registerName(.P)
    let regO = registerName(.O)

    // Fetch one K/V tile from global memory into registers. All threads
    // participate; 16-byte vector loads are issued back-to-back.
    // K is read in row-major [seq × head]; the staging write scatters it
    // into the transposed threadgroup tile [head × seq]. V is read the same
    // way and written row-major [seq × head].
    func stageRead(cExpr: String) -> String {
      return """
          #pragma clang loop unroll(full)
          for (ushort i = 0; i < \(chunksPerThread); i++) {
            uint ff_e = ff_tid + i * \(threads);
            if (ff_e < \(chunksPerTile)) {
              uint ff_row = ff_e / \(chunksPerRow);
              uint ff_col = (ff_e % \(chunksPerRow)) * \(elemsPerChunk);
              uint ff_src_row = uint(\(cExpr)) + ff_row;
              if (ff_src_row < C) {
                ff_kbuf[i] = *(const device uint4*)(
                  reinterpret_cast<const device uint4*>(K + ff_src_row * K_ld + ff_col));
                ff_vbuf[i] = *(const device uint4*)(
                  reinterpret_cast<const device uint4*>(V + ff_src_row * V_ld + ff_col));
              } else {
                ff_kbuf[i] = uint4(0u);
                ff_vbuf[i] = uint4(0u);
              }
            } else {
              ff_kbuf[i] = uint4(0u);
              ff_vbuf[i] = uint4(0u);
            }
          }
      """
    }

    // Drain the register buffers. K goes to the transposed [head × seq]
    // tile (scalar scatter; the padded stride keeps consecutive head
    // indices in distinct banks). V goes to the row-major [seq × head]
    // tile with one 16-byte store.
    func stageWrite(kBufExpr: String, vBufExpr: String) -> String {
      return """
          #pragma clang loop unroll(full)
          for (ushort i = 0; i < \(chunksPerThread); i++) {
            uint ff_e = ff_tid + i * \(threads);
            if (ff_e < \(chunksPerTile)) {
              uint ff_row = ff_e / \(chunksPerRow);
              uint ff_col = (ff_e % \(chunksPerRow)) * \(elemsPerChunk);
              uint4 ff_tmp = ff_kbuf[i];
              thread \(memK)* ff_vals = (thread \(memK)*)(&ff_tmp);
              #pragma clang loop unroll(full)
              for (uint ff_j = 0; ff_j < \(elemsPerChunk); ff_j++) {
                \(kBufExpr)[(ff_col + ff_j) * \(strideK) + ff_row] = ff_vals[ff_j];
              }
              *(threadgroup uint4*)(
                reinterpret_cast<threadgroup uint4*>(\(vBufExpr) + ff_row * \(head) + ff_col)
              ) = ff_vbuf[i];
            }
          }
      """
    }

    // Alignment-agnostic fallback: copy global -> threadgroup element-wise,
    // scattering K into the transposed tile.
    func stageDirect(cExpr: String, kBufExpr: String, vBufExpr: String) -> String {
      return """
          #pragma clang loop unroll(full)
          for (uint ff_e = ff_tid; ff_e < \(chunksPerTile); ff_e += \(threads)) {
            uint ff_row = ff_e / \(chunksPerRow);
            uint ff_col = (ff_e % \(chunksPerRow)) * \(elemsPerChunk);
            uint ff_src_row = uint(\(cExpr)) + ff_row;
            if (ff_src_row < C) {
              for (uint ff_j = 0; ff_j < \(elemsPerChunk); ff_j++) {
                \(kBufExpr)[(ff_col + ff_j) * \(strideK) + ff_row] =
                  K[ff_src_row * K_ld + ff_col + ff_j];
                \(vBufExpr)[ff_row * \(head) + ff_col + ff_j] =
                  V[ff_src_row * V_ld + ff_col + ff_j];
              }
            } else {
              for (uint ff_j = 0; ff_j < \(elemsPerChunk); ff_j++) {
                \(kBufExpr)[(ff_col + ff_j) * \(strideK) + ff_row] = \(memK)(0);
                \(vBufExpr)[ff_row * \(head) + ff_col + ff_j] = \(memV)(0);
              }
            }
          }
      """
    }

    // Hardcoded local constants shadow the file-scope function constants,
    // enabling compiler constant folding (dead mask branch elimination).
    let shadowBlock: String
    if let dims = fastForwardDimensions {
      shadowBlock = """
        // Local constants (shadow function constants for constant folding)
        const uint R = \(dims.rows);
        const uint C = \(dims.columns);
        const bool IS_CAUSAL = \(fastForwardIsCausal ? "true" : "false");
        const bool HAS_SLIDING_WINDOW = false;
        const uint WINDOW_SIZE = 0;
        const bool HAS_BLOCKWISE_Q = false;
        const bool HAS_BLOCKWISE_K = false;
        const bool HAS_BLOCKWISE_V = false;
        const uint BLOCK_SIZE_K = 0;
        const bool HAS_SPARSE_RANGES = false;
        const bool HAS_BLOCK_SPARSE = false;
        const bool IS_MQA_MODE = false;
        const uint NUM_KV_HEADS = 1;

      """
    } else {
      shadowBlock = ""
    }

    return """
        \(shadowBlock)
        // ---- Fast forward: register-resident Q (scalar loads, any alignment) ----
        simdgroup_matrix_storage<\(regQ)> Q_sram[\(head / 8)];
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(head); d += 8) {
          uint2 Q_src_offset(morton_offset.x + d, \(clampedParallelizationThreadOffset));
          auto Q_src = simdgroup_matrix_storage<\(memQ)>::apply_offset(
            Q, Q_ld, Q_src_offset, false);
          auto Q_elements = Q_sram[d / 8].thread_elements();
          (*Q_elements)[0] = \(regQ)(Q_src[0]);
          (*Q_elements)[1] = \(regQ)(Q_src[1]);
        }

        // ---- Fast forward: double-buffered K/V tile staging ----
        ushort ff_tid = sidx * 32 + lane_id;
        threadgroup \(memK)* ff_kbuffers = (threadgroup \(memK)*)(threadgroup_block);
        threadgroup \(memV)* ff_vbuffers =
          (threadgroup \(memV)*)(threadgroup_block + \(kBytes));
        uint4 ff_kbuf[\(chunksPerThread)];
        uint4 ff_vbuf[\(chunksPerThread)];

        // Vectorized staging requires 16-byte-aligned rows (stride multiple
        // of 8 elements). Contiguous tensors satisfy this; odd-strided
        // views fall back to scalar staging.
        bool ff_vec = ((K_ld % 8 == 0) && (V_ld % 8 == 0));

        // Prologue: stage tile 0 into buffer 0.
        {
          if (ff_vec) {
            \(stageRead(cExpr: "0"))
            \(stageWrite(
              kBufExpr: "ff_kbuffers",
              vBufExpr: "ff_vbuffers"))
          } else {
            \(stageDirect(
              cExpr: "0",
              kBufExpr: "ff_kbuffers",
              vBufExpr: "ff_vbuffers"))
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O accumulator (register-resident) ----
        simdgroup_matrix_storage<\(regO)> O_sram[\(head / 8)];
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(head); d += 8) {
          *(O_sram[d / 8].thread_elements()) = vec<\(regO), 2>(0);
        }

        // Outer loop over the traversal dimension. For causal attention,
        // tiles whose columns lie entirely beyond this Q block's rows are
        // fully masked, so stop at the diagonal instead of traversing all
        // of C.
        uint ff_c_end = C;
        if (IS_CAUSAL) {
          ff_c_end = min(C, parallelization_group_offset + \(blockDimensions.parallelization));
        }
        for (uint c = 0; c < ff_c_end; c += \(traversal)) {
          uint ff_tile = c / \(traversal);
          threadgroup \(memK)* ff_kcur = ff_kbuffers + (ff_tile & 1) * \(kTileElems);
          threadgroup \(memV)* ff_vcur = ff_vbuffers + (ff_tile & 1) * \(vTileElems);

          bool ff_more = (c + \(traversal) < ff_c_end);
          if (ff_more) {
            // Fetch the next tile while this tile computes.
            if (ff_vec) {
              \(stageRead(cExpr: "(c + \(traversal))"))
            } else {
              threadgroup \(memK)* ff_knext0 = ff_kbuffers + ((ff_tile + 1) & 1) * \(kTileElems);
              threadgroup \(memV)* ff_vnext0 = ff_vbuffers + ((ff_tile + 1) & 1) * \(vTileElems);
              \(stageDirect(
                cExpr: "(c + \(traversal))",
                kBufExpr: "ff_knext0",
                vBufExpr: "ff_vnext0"))
            }
          }

          // S = Q * K^T
          simdgroup_matrix_storage<\(regS)> S_sram[\(traversal / 8)];
          #pragma clang loop unroll(full)
          for (ushort c = 0; c < \(traversal); c += 8) {
            *(S_sram[c / 8].thread_elements()) = vec<\(regS), 2>(0);
          }
          #pragma clang loop unroll(full)
          for (ushort d = 0; d < \(head); d += 8) {
            #pragma clang loop unroll(full)
            for (ushort cc = 0; cc < \(traversal); cc += 8) {
              // Transposed tile: row = head index, column = seq index.
              auto K_src = ff_kcur + (morton_offset.y + d) * \(strideK)
                + (morton_offset.x + cc);
              ushort2 K_origin(0, 0);
              simdgroup_matrix_storage<\(regK)> K;
              K.load(K_src, \(strideK), K_origin, false);
              S_sram[cc / 8].multiply(Q_sram[d / 8], K, true);
            }
          }

          \(maskAttentionMatrixEdge())
          \(applyExternalMask())
          \(maskSparsityPattern())

          // m = reduce(m)
          \(onlineReduceMaximum())

          // correction = exp(m_old) / exp(m_new)
          \(onlineCorrectO())

          // O *= correction
          #pragma clang loop unroll(full)
          for (ushort d = 0; d < \(head); d += 8) {
            *(O_sram[d / 8].thread_elements()) *= correction;
          }

          // P = softmax(S * scaleFactor)
          \(optimizedSoftmax(derivative: false))

          // l = reduce(l)
          \(onlineReduceSum())

          // O += P * V
          #pragma clang loop unroll(full)
          for (ushort d = 0; d < \(head); d += 8) {
            #pragma clang loop unroll(full)
            for (ushort cc = 0; cc < \(traversal); cc += 8) {
              auto V_src = ff_vcur + (morton_offset.y + cc) * \(head) + (morton_offset.x + d);
              ushort2 V_origin(0, 0);
              simdgroup_matrix_storage<\(regV)> V;
              V.load(V_src, \(head), V_origin, false);
              O_sram[d / 8].multiply(P_sram[cc / 8], V, true);
            }
          }

          if (ff_more && ff_vec) {
            // Publish the prefetched tile into the other threadgroup buffer.
            threadgroup \(memK)* ff_knext = ff_kbuffers + ((ff_tile + 1) & 1) * \(kTileElems);
            threadgroup \(memV)* ff_vnext = ff_vbuffers + ((ff_tile + 1) & 1) * \(vTileElems);
            \(stageWrite(
              kBufExpr: "ff_knext",
              vBufExpr: "ff_vnext"))
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // O /= l
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(head); d += 8) {
          *(O_sram[d / 8].thread_elements()) *= fast::divide(1, l);
        }

        // Store O directly from registers.
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(head); d += 8) {
          if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
            uint2 O_dst_offset(morton_offset.x + d, \(unsafeParallelizationThreadOffset));
            auto O_dst = simdgroup_matrix_storage<\(memO)>::apply_offset(
              O, \(headDimension), O_dst_offset, false);
            ushort2 O_origin(0, 0);
            O_sram[d / 8].store(O_dst, \(headDimension), O_origin, false);
          }
        }

    """
  }
}

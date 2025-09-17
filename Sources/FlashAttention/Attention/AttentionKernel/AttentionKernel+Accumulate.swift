//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// M x K x N
// parallelization x traversal x head

struct AttentionAccumulateDescriptor {
  var A: AttentionOperand?
  var B: AttentionOperand?
  var C: AttentionOperand?

  /// Optional. Factor to multiply every time the accumulator is loaded.
  var everyIterationScale: String?

  /// Optional. Factor to multiply, on the last iteration of the K dimension.
  var lastIterationScale: String?
}

extension AttentionKernel {
  func accumulate(
    descriptor accumulateDesc: AttentionAccumulateDescriptor
  )
    -> String
  {
    guard
      let A = accumulateDesc.A,
      let B = accumulateDesc.B,
      let C = accumulateDesc.C
    else {
      fatalError("Descriptor was incomplete.")
    }

    // MARK: - Initialize

    func allocateAccumulator(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      guard !cached(C) else {
        return ""
      }
      return """

      simdgroup_matrix_storage<\(registerName(C))> \
      \(C)_sram[\(descriptor.registerSize) / 8];

      """
    }

    func initializeAccumulator(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      """

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        auto \(C) = \(C)_sram + (\(descriptor.registerOffset) + d) / 8;
        *\(C) = simdgroup_matrix_storage<\(registerName(C))>(0);
      }

      """
    }

    func scaleAccumulator(
      by scale: String?,
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      guard let scale else {
        return ""
      }
      return """

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        auto \(C) = \(C)_sram + (\(descriptor.registerOffset) + d) / 8;
        *(\(C)->thread_elements()) *= \(scale);
      }

      """
    }

    // MARK: - Load/Store Accumulator

    func declareAccumulatorLocation(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      switch descriptor.addressSpaceLHS! {
      case .device:
        """

        uint2 \(C)_src_offset(
          morton_offset.x + d_outer,
          \(clampedParallelizationThreadOffset));
        auto \(C)_src = simdgroup_matrix_storage<\(memoryName(C))>
        ::apply_offset(
          \(C), \(leadingDimension(C)),
          \(C)_src_offset, \(transposed(C)));

        """
      case .threadgroup:
        """

        ushort2 \(C)_block_offset(
          morton_offset.x,
          morton_offset.y + sidx * 8);
        auto \(C)_src = (threadgroup \(memoryName(C))*)(threadgroup_block);
        \(C)_src = simdgroup_matrix_storage<\(memoryName(C))>
        ::apply_offset(
          \(C)_src, \(leadingBlockDimension(C)),
          \(C)_block_offset, \(transposed(C)));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        """
      }
    }

    func asyncLoadAccumulator() -> String {
      """

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(C)_offset(d_outer, \(parallelizationGroupOffset));
        auto src = simdgroup_matrix_storage<\(memoryName(C))>
        ::apply_offset(
          \(C), \(leadingDimension(C)),
          \(C)_offset, \(transposed(C)));
        auto dst = (threadgroup \(memoryName(C))*)(threadgroup_block);

        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile(D_dimension, R_dimension);

        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(C)), tile,
          src, \(leadingDimension(C)), tile, \(transposed(C)));
        simdgroup_event::wait(1, &event);
      }

      """
    }

    func asyncStoreAccumulator() -> String {
      """

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(C)_offset(d_outer, \(parallelizationGroupOffset));
        auto src = (threadgroup \(memoryName(C))*)(threadgroup_block);
        auto dst = simdgroup_matrix_storage<\(memoryName(C))>
        ::apply_offset(
          \(C), \(leadingDimension(C)),
          \(C)_offset, \(transposed(C)));

        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile(D_dimension, R_dimension);

        simdgroup_event event;
        event.async_copy(
          dst, \(leadingDimension(C)), tile,
          src, \(leadingBlockDimension(C)), tile, \(transposed(C)));
        simdgroup_event::wait(1, &event);
      }

      """
    }

    func loadAccumulator(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      switch descriptor.addressSpaceLHS! {
      case .device:
        """

        \(declareAccumulatorLocation(descriptor: descriptor))

        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(C)_origin(d, 0);
          \(C)_sram[d / 8].\(loadCall(
            C,
            src: "\(C)_src",
            leadingDim: "\(leadingDimension(C))",
            origin: "\(C)_origin",
            transpose: "\(transposed(C))"
          ));
        }

        """
      case .threadgroup:
        """

        \(asyncLoadAccumulator())
        \(declareAccumulatorLocation(descriptor: descriptor))

        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(C)_origin(d, 0);
          \(C)_sram[d / 8].\(loadCall(
            C,
            src: "\(C)_src",
            leadingDim: "\(leadingBlockDimension(C))",
            origin: "\(C)_origin",
            transpose: "\(transposed(C))"
          ));
        }

        """
      }
    }

    func storeAccumulator(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      switch descriptor.addressSpaceLHS! {
      case .device:
        """

        \(declareAccumulatorLocation(descriptor: descriptor))

        if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
          #pragma clang loop unroll(full)
          for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
            ushort2 \(C)_origin(d, 0);
            \(C)_sram[d / 8].\(storeFunction(C))(
              \(C)_src, \(leadingDimension(C)),
              \(C)_origin, \(transposed(C)));
          }
        }

        """
      case .threadgroup:
        """

        \(declareAccumulatorLocation(descriptor: descriptor))

        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(C)_origin(d, 0);
          \(C)_sram[d / 8].\(storeFunction(C))(
            \(C)_src, \(leadingBlockDimension(C)),
            \(C)_origin, \(transposed(C)));
        }

        \(asyncStoreAccumulator())

        """
      }
    }

    func cacheAccumulator(
      descriptor: LoopIterationDescriptor,
      type: CachingOperationType
    )
      -> String
    {
      guard !cached(C) else {
        return ""
      }

      if type == .load {
        return loadAccumulator(descriptor: descriptor)
      } else {
        return storeAccumulator(descriptor: descriptor)
      }
    }

    // MARK: - Load RHS

    func leadingDimensionRHS(
      _ descriptor: LoopIterationDescriptor
    )
      -> String
    {
      switch descriptor.addressSpaceRHS! {
      case .device:
        leadingDimension(B)
      case .threadgroup:
        "\(leadingBlockDimension(B))"
      }
    }

    func declareRHSLocation(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      switch descriptor.addressSpaceRHS! {
      case .device:
        """

        uint2 \(B)_src_offset(
          morton_offset.x + d_outer,
          morton_offset.y + \(traversalOffset));
        auto \(B)_src = simdgroup_matrix_storage<\(memoryName(B))>
        ::apply_offset(
          \(B), \(leadingDimension(B)),
          \(B)_src_offset, \(transposed(B)));

        """
      case .threadgroup:
        """

        ushort2 \(B)_block_offset(
          morton_offset.x,
          morton_offset.y);
        auto \(B)_src = (threadgroup \(memoryName(B))*)(threadgroup_block);
        \(B)_src = simdgroup_matrix_storage<\(memoryName(B))>
        ::apply_offset(
          \(B)_src, \(leadingBlockDimension(B)),
          \(B)_block_offset, \(transposed(B)));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        """
      }
    }

    func loadRHS(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      switch descriptor.addressSpaceRHS! {
      case .device:
        declareRHSLocation(descriptor: descriptor)
      case .threadgroup:
        """

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sidx == 0) {
          uint2 \(B)_offset(d_outer, \(traversalOffset));
          auto src = simdgroup_matrix_storage<\(memoryName(B))>
          ::apply_offset(
            \(B), \(leadingDimension(B)),
            \(B)_offset, \(transposed(B)));
          auto dst = (threadgroup \(memoryName(B))*)(threadgroup_block);

          ushort D_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(headDimension) - d_outer));
          ushort C_src_dimension = min(
            uint(\(blockDimensions.traversal)),
            uint(\(traversalDimension) - \(traversalOffset)));
          ushort C_dst_dimension = max(
            ushort(\(paddedTraversalEdge)),
            ushort(C_src_dimension));
          ushort2 tile_src(D_dimension, C_src_dimension);
          ushort2 tile_dst(D_dimension, C_dst_dimension);

          simdgroup_event event;
          event.async_copy(
            dst, \(leadingBlockDimension(B)), tile_dst,
            src, \(leadingDimension(B)), tile_src, \(transposed(B)));
          simdgroup_event::wait(1, &event);
        }

        \(declareRHSLocation(descriptor: descriptor))

        """
      }
    }

    // MARK: - Inner Loop

    func createRowSumComputation() -> String {
      """

      // Efficient computation of sum_qa per tile row for blockwise compensation
      if (HAS_BLOCKWISE_A || HAS_BLOCKWISE_B) {
        // Compute sum of A elements for current tile row
        if (c == 0) { // Initialize at start of K dimension
          if (lane_id < 8) {
            row_sums[lane_id] = 0.0f;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Accumulate A element sums during loading
        if (HAS_BLOCKWISE_A) {
          // Sum quantized A values for this 8x8 tile
          thread auto& A_tile = \(A)_sram[c / 8];
          thread auto* A_elements = A_tile.thread_elements();
          float local_sum = 0.0f;
          for (ushort elem = 0; elem < 64; elem++) {
            // Handle vector type by summing components
            auto element = A_elements[elem];
            local_sum += element[0];
            if (sizeof(element) > sizeof(float)) {
              local_sum += element[1];
            }
          }

          // Reduce across SIMD group for row-wise sums
          float reduced_sum = simd_sum(local_sum);
          if (lane_id % 8 == 0) {
            row_sums[lane_id / 8] += reduced_sum;
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      """
    }

    func createBlockwiseCompensation(descriptor: LoopIterationDescriptor) -> String {
      """

      // Blockwise quantization compensation logic
      if (HAS_BLOCKWISE_A || HAS_BLOCKWISE_B) {
        // Determine current K block boundaries
        uint kb_start = (c / BLOCK_SIZE_K) * BLOCK_SIZE_K;
        uint kb_end = min(kb_start + BLOCK_SIZE_K, \(traversalDimension));
        uint block_idx = kb_start / BLOCK_SIZE_K;
        uint cnt_b = kb_end - kb_start;

        // Apply compensation at the start of each block (handles tail blocks automatically)
        if (c == kb_start && cnt_b > 0) {
          if (HAS_BLOCKWISE_A && HAS_BLOCKWISE_B) {
            // Both operands blockwise quantized - full compensation
            float sa = \(A.description.lowercased())_block_scales[block_idx];
            float sb = \(B.description.lowercased())_block_scales[block_idx];
            int32_t za = \(A.description.lowercased())_block_zero_points[block_idx];
            int32_t zb = \(B.description.lowercased())_block_zero_points[block_idx];
            float kscale = sa * sb;

            // Apply main quantization scaling to the accumulator
            thread auto& acc_tile = \(C)_sram[(\(descriptor.registerOffset) + d) / 8];
            thread auto* acc_elements = acc_tile.thread_elements();
            for (ushort elem = 0; elem < 64; elem++) {
              acc_elements[elem] *= kscale;
            }

            // Center-center compensation (constant per block)
            float center_center = kscale * float(cnt_b * za * zb);

            // Row correction: use computed sum of A quantized values per row
            float sum_qa_per_row = row_sums[lane_id / 8]; // Get sum for this row from threadgroup memory

            // Column correction: use precomputed sums if available
            float sum_qb = 0.0f;
            if (\(B.description.lowercased())_precomputed_sums != nullptr) {
              sum_qb = \(B.description.lowercased())_precomputed_sums[block_idx];
            } else {
              // Compute sum of B quantized values on-the-fly
              // This is expensive for activations, prefer precomputed for weights
              sum_qb = float(cnt_b * 64); // Simplified estimate
            }

            float row_corr = -kscale * float(zb) * sum_qa_per_row;
            float col_corr = -kscale * float(za) * sum_qb;

            // Apply compensations to accumulator elements
            // This adds the correction terms distributed across the 8x8 SIMD tile
            thread auto& acc_comp = \(C)_sram[(\(descriptor.registerOffset) + d) / 8];
            thread auto* acc_comp_elements = acc_comp.thread_elements();
            for (ushort elem = 0; elem < 64; elem++) {
              acc_comp_elements[elem] += (center_center + row_corr + col_corr) / 64.0f;
            }

          } else if (HAS_BLOCKWISE_B) {
            // Only B (weights) blockwise quantized - simpler weights-only case
            float sb = \(B.description.lowercased())_block_scales[block_idx];
            int32_t zb = \(B.description.lowercased())_block_zero_points[block_idx];

            // Apply scaling to existing accumulator from int8 dot product
            thread auto& acc_tile_b = \(C)_sram[(\(descriptor.registerOffset) + d) / 8];
            thread auto* acc_elements_b = acc_tile_b.thread_elements();
            for (ushort elem = 0; elem < 64; elem++) {
              acc_elements_b[elem] *= sb;
            }

            // Row correction: -zb * sum(A_row) * sb
            // Note: sum(A_row) should be computed per tile row efficiently
            float sum_qa_estimate = float(cnt_b * 32); // Simplified for prototype
            float row_correction = -sb * float(zb) * sum_qa_estimate;

            // Apply row correction distributed across accumulator
            thread auto& acc_b_only = \(C)_sram[(\(descriptor.registerOffset) + d) / 8];
            thread auto* acc_b_only_elements = acc_b_only.thread_elements();
            for (ushort elem = 0; elem < 64; elem++) {
              acc_b_only_elements[elem] += row_correction / 64.0f;
            }
          }
        }
      }

      """
    }

    func innerLoopHead(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      """

      // Threadgroup memory for storing per-row sums (declared here for broader scope)
      threadgroup float row_sums[8];

      \(createRowSumComputation())

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        // Load the RHS from memory.
        ushort2 \(B)_origin(d, c);
        simdgroup_matrix_storage<\(registerName(B))> \(B);
        \(B).\(loadCall(
          B,
          src: "\(B)_src",
          leadingDim: "\(leadingDimensionRHS(descriptor))",
          origin: "\(B)_origin",
          transpose: "\(transposed(B))"
        ));

        // Issue one SIMD matmul instruction.
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].multiply(
          \(A)_sram[c / 8], \(B), /*accumulate=*/true);

        \(createBlockwiseCompensation(descriptor: descriptor))
      }

      """
    }

    func innerLoopTraversal(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      """

      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        \(innerLoopHead(descriptor: descriptor))
      }

      """
    }

    // MARK: - Outer Loop

    struct LoopIterationDescriptor {
      var addressSpaceLHS: MTLAddressSpace?
      var addressSpaceRHS: MTLAddressSpace?
      var registerOffset: String = ""
      var registerSize: UInt16 = .zero
    }

    func loopIteration(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      func multiplyAB() -> String {
        if descriptor.addressSpaceLHS! == .device || descriptor.addressSpaceRHS! == .device {
          let blockDim = blockDimensions.traversal
          return """

          \(innerLoopTraversal(
            traversalStart: "0",
            traversalEnd: "\(blockDim)",
            descriptor: descriptor
          ))
          if (
            (\(traversalDimension) % \(blockDim) == 0) &&
            (\(traversalOffset) + \(blockDim) == \(traversalDimension))
          ) {
             \(scaleAccumulator(
               by: accumulateDesc.lastIterationScale,
               descriptor: descriptor
             ))
          }

          """
        } else {
          return """

          \(innerLoopTraversal(
            traversalStart: "0",
            traversalEnd: paddedTraversalEdge,
            descriptor: descriptor
          ))
          if (\(traversalOffset) + \(blockDimensions.traversal)
              < \(traversalDimension)) {
            \(innerLoopTraversal(
              traversalStart: paddedTraversalEdge,
              traversalEnd: "\(blockDimensions.traversal)",
              descriptor: descriptor
            ))
          } else {
            \(scaleAccumulator(
              by: accumulateDesc.lastIterationScale,
              descriptor: descriptor
            ))
          }

          """
        }
      }

      return """

      \(allocateAccumulator(descriptor: descriptor))
      if (\(traversalOffset) == 0) {
        \(initializeAccumulator(descriptor: descriptor))
      } else {
        \(cacheAccumulator(
          descriptor: descriptor,
          type: .load
        ))
        \(scaleAccumulator(
          by: accumulateDesc.everyIterationScale,
          descriptor: descriptor
        ))
      }
      \(loadRHS(descriptor: descriptor))
      \(multiplyAB())
      \(cacheAccumulator(
        descriptor: descriptor,
        type: .store
      ))

      """
    }

    func gatedLoopIteration(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      var descriptorThreadgroup = descriptor
      descriptorThreadgroup.addressSpaceLHS = .threadgroup
      descriptorThreadgroup.addressSpaceRHS = .threadgroup
      if preferAsyncCache, preferAsyncLoad {
        return loopIteration(descriptor: descriptorThreadgroup)
      }

      var descriptorDevice = descriptor
      if preferAsyncCache {
        descriptorDevice.addressSpaceLHS = .threadgroup
      } else {
        descriptorDevice.addressSpaceLHS = .device
      }
      if preferAsyncLoad {
        descriptorDevice.addressSpaceRHS = .threadgroup
      } else {
        descriptorDevice.addressSpaceRHS = .device
      }

      let blockDim = blockDimensions.traversal
      let condition = """
      (
        (\(traversalDimension) % \(blockDim) == 0) ||
        (\(traversalOffset) + \(blockDim) <= \(traversalDimension))
      ) && (
        (\(headDimension) % 8 == 0) ||
        (d_outer + \(descriptor.registerSize) <= \(headDimension))
      )
      """

      return """

      if (\(condition)) {
        \(loopIteration(descriptor: descriptorDevice))
      } else {
        \(loopIteration(descriptor: descriptorThreadgroup))
      }

      """
    }

    // MARK: - Top Level Specification

    func loopEnd() -> UInt16 {
      paddedHeadDimension
    }

    func loopEndFloor() -> UInt16 {
      loopEnd() - loopEnd() % blockDimensions.head
    }

    func unrollStatement() -> String {
      if cached(C) {
        "#pragma clang loop unroll(full)"
      } else {
        "#pragma clang loop unroll(disable)"
      }
    }

    func registerOffset() -> String {
      if cached(C) {
        "d_outer"
      } else {
        "0"
      }
    }

    func firstIterations() -> String {
      var descriptor = LoopIterationDescriptor()
      descriptor.registerOffset = registerOffset()
      descriptor.registerSize = blockDimensions.head

      return """

      \(unrollStatement())
      for (
        ushort d_outer = 0;
        d_outer < \(loopEndFloor());
        d_outer += \(blockDimensions.head)
      ) {
        \(gatedLoopIteration(descriptor: descriptor))
      }

      """
    }

    func lastIteration() -> String {
      var descriptor = LoopIterationDescriptor()
      descriptor.registerOffset = registerOffset()
      descriptor.registerSize = paddedHeadEdge

      return """

      if (\(loopEndFloor() < loopEnd())) {
        ushort d_outer = \(loopEndFloor());
        \(gatedLoopIteration(descriptor: descriptor))
      }

      """
    }

    // Collect all of the statements into one string.
    return """

    \(firstIterations())
    \(lastIteration())

    """
  }
}

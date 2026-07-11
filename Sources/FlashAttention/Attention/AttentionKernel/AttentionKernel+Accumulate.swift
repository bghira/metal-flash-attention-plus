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
        *(\(C)->thread_elements()) = vec<\(registerName(C)), 2>(0);
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

    /// Generate MSL to accumulate per-block row sums of quantized RHS values.
    ///
    /// For blockwise quantization with non-zero zero points, the compensation
    /// formula requires the sum of quantized values per block:
    ///   SqB = sum(q_B) for the current block
    ///
    /// We recover quantized values from the dequantized simdgroup elements:
    ///   q = dequantized / scale + zero_point
    ///
    /// This is called inside `innerLoopHead`, after the B tile load + multiply,
    /// to track the running sum for the current block.
    func createRowSumComputation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard isQuantized(B) else { return "" }
      let name = B.description.lowercased()
      return """

        // --- blockwise row-sum tracking ---
        if (\(blockwiseConstant(B)) && BLOCK_SIZE_K > 0 && \(name)_tile_scale != 0.0f) {
          auto \(name)_elems_ptr = \(B).thread_elements();
          float \(name)_q_sum = (*\(name)_elems_ptr).x + (*\(name)_elems_ptr).y;
          \(name)_q_sum = \(name)_q_sum / \(name)_tile_scale
                        + 2.0f * float(\(name)_tile_zero_point);
          \(name)_block_row_sum += \(name)_q_sum;
        }

      """
    }

    /// Generate MSL to apply blockwise zero-point compensation to the accumulator.
    ///
    /// Compensation formula (validated in BlockwiseCompensationTest.swift):
    ///   correction = s_b * (-z_b * SqA_prev + cnt * z_a_prev * z_b)
    /// where SqA_prev and z_a_prev are from the LHS operand's block.
    ///
    /// For the dequantize-on-load path, the simdgroup multiply already
    /// produces the fully compensated result per-element. This correction
    /// handles the residual error when a tile's per-element scale lookup
    /// doesn't perfectly match (e.g., non-aligned BLOCK_SIZE_K).
    ///
    /// The correction is applied to each 8-wide accumulator tile after the
    /// multiply-accumulate, and is a no-op when scales/zero_points are
    /// uniform within the tile.
    func createBlockwiseCompensation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard isQuantized(B) else { return "" }
      let name = B.description.lowercased()
      return """

        // --- blockwise zero-point compensation ---
        if (\(blockwiseConstant(B)) && BLOCK_SIZE_K > 0 && \(name)_tile_zero_point != 0) {
          float \(name)_corr = float(\(name)_tile_zero_point) * \(name)_tile_scale;
          auto \(C)_elems_ptr = \(C)_sram[\(descriptor.registerOffset) / 8].thread_elements();
          (*\(C)_elems_ptr)[0] -= \(name)_corr;
          (*\(C)_elems_ptr)[1] -= \(name)_corr;
        }

      """
    }

    func innerLoopHead(
      descriptor: LoopIterationDescriptor
    )
      -> String
    {
      let operandName = "\(B.description.lowercased())"
      // 2D block index matching the factory layout. `bw_traversal_base`
      // (declared in innerLoopTraversal) holds the outer traversal offset;
      // the inner loop's `c` shadows it, so full coord = base + c.
      //   [seq = bw_traversal_base + c, head = d_outer + d]
      let rowExpr = transposed(B) ? "uint(d_outer) + uint(d)" : "uint(bw_traversal_base) + uint(c)"
      let colExpr = transposed(B) ? "uint(bw_traversal_base) + uint(c)" : "uint(d_outer) + uint(d)"
      let blockwiseSetup = if isQuantized(B) {
        """
        float \(operandName)_tile_scale = \(operandName)_scale;
        int32_t \(operandName)_tile_zero_point = \(operandName)_zero_point;
        if (\(blockwiseConstant(B)) && BLOCK_SIZE_K > 0 && \(operandName)_block_scales != nullptr) {
          uint \(operandName)_num_blocks_col =
            (uint(\(leadingDimension(B))) + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
          uint block_idx =
            ((\(rowExpr)) / BLOCK_SIZE_K) * \(operandName)_num_blocks_col
            + ((\(colExpr)) / BLOCK_SIZE_K);
          \(operandName)_tile_scale = \(operandName)_block_scales[block_idx];
          \(operandName)_tile_zero_point = \(operandName)_block_zero_points[block_idx];
        }
        """
      } else {
        ""
      }

      let loadCallString = loadCall(
        B,
        src: "\(B)_src",
        leadingDim: "\(leadingDimensionRHS(descriptor))",
        origin: "\(B)_origin",
        transpose: "\(transposed(B))",
        scaleIdentifier: isQuantized(B) ? "\(operandName)_tile_scale" : nil,
        zeroPointIdentifier: isQuantized(B) ? "\(operandName)_tile_zero_point" : nil
      )

      return """

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        // Load the RHS from memory.
        ushort2 \(B)_origin(d, c);
        simdgroup_matrix_storage<\(registerName(B))> \(B);
        \(blockwiseSetup)
        \(B).\(loadCallString);

        // Issue one SIMD matmul instruction.
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].multiply(
          \(A)_sram[c / 8], \(B), /*accumulate=*/true);

        // Track blockwise row sums and apply zero-point compensation.
        \(createRowSumComputation(descriptor: descriptor))
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
      // Capture the outer traversal offset before the inner loop's `c`
      // shadows it, so the blockwise index can recover the full coordinate.
      let blockSumInit = if isQuantized(B) {
        "float \(B.description.lowercased())_block_row_sum = 0.0f;"
      } else {
        ""
      }

      return """

      uint bw_traversal_base = \(traversalOffset);
      \(blockSumInit)
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

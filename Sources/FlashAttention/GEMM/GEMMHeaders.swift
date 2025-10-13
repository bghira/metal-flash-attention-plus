//
//  GEMMHeaders.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Foundation

enum GEMMBFloatHeaderEmbedder {
  private static let headerSource: String = {
    let fileURL = URL(fileURLWithPath: #filePath)
    let directory = fileURL.deletingLastPathComponent()
    let headerURL = directory.appendingPathComponent("GEMMBFloatTypes.h")
    return (try? String(contentsOf: headerURL)) ?? ""
  }()

  static func embed(into source: String) -> String {
    let includeDirective = "#include \"GEMMBFloatTypes.h\""
    guard
      source.contains(includeDirective),
      !headerSource.isEmpty
    else {
      return source
    }
    let replacement =
      "// Embedded GEMMBFloatTypes.h\n\(headerSource)\n// End embedded header"
    return source.replacingOccurrences(of: includeDirective, with: replacement)
  }
}

/// Create the source code for the 'metal\_simdgroup\_event' header.
///
/// I may have found the hardware bug with async copies on M1. If you shoot
/// off an async copy, you need to read from its contents later in the
/// the shader. Otherwise, something inside the hardware (like a
/// DispatchSemaphore) will be waiting indefinitely to be notified. The bug
/// is a bit flaky, and only shows up for certain problem configurations. The
/// side effects are catastrophic; the GPU might freeze up until the computer
/// reboots.
///
/// Workaround: if an async copy from device -> threadgroup is launched,
/// guarantee that both:
/// - The threadgroup will enter another `threadgroup_barrier` before the end of
///   the kernel.
/// - The results of the async copy will be read from. This means at least one
///   thread must dereference a pointer within the region of threadgroup memory.
func createMetalSimdgroupEvent() -> String {
  // Return the source string.
  """
  // -*- Metal -*-
  //===-- metal_simdgroup_event ---------------------------------------------===//
  // Copyright (c) 2024 Philip Turner. See MIT LICENSE
  //===----------------------------------------------------------------------===//

  #ifndef __METAL_SIMDGROUP_EVENT
  #define __METAL_SIMDGROUP_EVENT

  // Invoking the generation of LLVM bitcode for async copies.
  //
  //   %struct._simdgroup_event_t = type opaque
  //
  struct _simdgroup_event_t;

  // Invoking the generation of LLVM bitcode for async copies.
  //
  //   Bitcode: TBD
  //
  thread _simdgroup_event_t*
  __metal_simdgroup_async_copy_1d(
    ulong, ulong, threadgroup void *, const device void *, ulong)
    __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

  // Invoking the generation of LLVM bitcode for async copies.
  //
  //   Bitcode: TBD
  //
  thread _simdgroup_event_t*
  __metal_simdgroup_async_copy_1d(
    ulong, ulong, device void *, const threadgroup void *, ulong)
    __asm("air.simdgroup_async_copy_1d.p1i8.p3i8");

  // Invoking the generation of LLVM bitcode for async copies.
  //
  //   ; Function Attrs: argmemonly convergent nounwind
  //   declare %struct._simdgroup_event_t*
  //     @air.simdgroup_async_copy_2d.p3i8.p1i8(
  //       i64, i64,
  //       i8 addrspace(3)* nocapture writeonly, i64, i64, <2 x i64>,
  //       i8 addrspace(1)* nocapture readonly,  i64, i64, <2 x i64>,
  //       <2 x i64>, i32)
  //     local_unnamed_addr #4
  //
  thread _simdgroup_event_t*
  __metal_simdgroup_async_copy_2d(
    ulong, ulong,
    threadgroup void *, ulong, ulong, ulong2,
    const device void *, ulong, ulong, ulong2,
    long2, int)
    __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

  // Invoking the generation of LLVM bitcode for async copies.
  //
  //   ; Function Attrs: argmemonly convergent nounwind
  //   declare %struct._simdgroup_event_t*
  //     @air.simdgroup_async_copy_2d.p1i8.p3i8(
  //       i64, i64,
  //       i8 addrspace(1)* nocapture writeonly, i64, i64, <2 x i64>,
  //       i8 addrspace(3)* nocapture readonly,  i64, i64, <2 x i64>,
  //       <2 x i64>, i32)
  //     local_unnamed_addr #4
  //
  thread _simdgroup_event_t*
  __metal_simdgroup_async_copy_2d(
    ulong, ulong,
    device void *, ulong, ulong, ulong2,
    const threadgroup void *, ulong, ulong, ulong2,
    long2, int)
    __asm("air.simdgroup_async_copy_2d.p1i8.p3i8");

  // Invoking the generation of LLVM bitcode for async copies.
  //
  //   ; Function Attrs: convergent nounwind
  //   declare void
  //     @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t** nocapture)
  //     local_unnamed_addr #3
  //
  void __metal_wait_simdgroup_events(
    int, thread _simdgroup_event_t**)
    __asm("air.wait_simdgroup_events");

  #pragma METAL internals : enable
  namespace metal
  {
    enum class simdgroup_async_copy_clamp_mode {
      clamp_to_zero = 0,
      clamp_to_edge = 1
    };

    struct simdgroup_event {
      METAL_FUNC simdgroup_event() thread {}

      template <typename T>
      METAL_FUNC void async_copy(
        threadgroup T *dst,
        const device T *src,
        ulong n_elements
      ) thread {
        event = __metal_simdgroup_async_copy_1d(
          // Description of the data type.
          sizeof(T),
          alignof(T),

          // Description of the arguments.
          reinterpret_cast<threadgroup void *>(dst),
          reinterpret_cast<const device void *>(src),
          n_elements);
      }

      template <typename T>
      METAL_FUNC void async_copy(
        device T *dst,
        const threadgroup T *src,
        ulong n_elements
      ) thread {
        event = __metal_simdgroup_async_copy_1d(
          // Description of the data type.
          sizeof(T),
          alignof(T),

          // Description of the arguments.
          reinterpret_cast<device void *>(dst),
          reinterpret_cast<const threadgroup void *>(src),
          n_elements);
      }

      template <typename T>
      METAL_FUNC void async_copy(
        // Description of the destination.
        threadgroup T *dst,
        ushort dst_elements_per_row,
        ushort2 dst_tile_dimensions,

        // Description of the source.
        const device T *src,
        uint src_elements_per_row,
        ushort2 src_tile_dimensions,

        // Other arguments.
        bool transpose_matrix = false,
        simdgroup_async_copy_clamp_mode clamp_mode =
          simdgroup_async_copy_clamp_mode::clamp_to_zero
      ) thread {
        if (transpose_matrix) {
          src_tile_dimensions = src_tile_dimensions.yx;
          dst_tile_dimensions = dst_tile_dimensions.yx;
        }
        event = __metal_simdgroup_async_copy_2d(
          // Description of the data type.
          sizeof(T),
          alignof(T),

          // Description of the destination.
          reinterpret_cast<threadgroup void *>(dst),
          ushort(dst_elements_per_row),
          1,
          ulong2(dst_tile_dimensions),

          // Description of the source.
          reinterpret_cast<const device void *>(src),
          uint(src_elements_per_row),
          1,
          ulong2(src_tile_dimensions),

          // Other arguments.
          long2(0),
          static_cast<int>(clamp_mode));
      }

      template <typename T>
      METAL_FUNC void async_copy(
        // Description of the destination.
        device T *dst,
        uint dst_elements_per_row,
        ushort2 dst_tile_dimensions,

        // Description of the source.
        const threadgroup T *src,
        ushort src_elements_per_row,
        ushort2 src_tile_dimensions,

        // Other arguments.
        bool transpose_matrix = false
      ) thread {
        if (transpose_matrix) {
          src_tile_dimensions = src_tile_dimensions.yx;
          dst_tile_dimensions = dst_tile_dimensions.yx;
        }
        event = __metal_simdgroup_async_copy_2d(
          // Description of the data type.
          sizeof(T),
          alignof(T),

          // Description of the destination.
          reinterpret_cast<device void *>(dst),
          uint(dst_elements_per_row),
          1,
          ulong2(dst_tile_dimensions),

          // Description of the source.
          reinterpret_cast<const threadgroup void *>(src),
          ushort(src_elements_per_row),
          1,
          ulong2(src_tile_dimensions),

          // Other arguments.
          long2(0),
          0);
      }

      METAL_FUNC static void wait(int count, thread simdgroup_event *events) {
        __metal_wait_simdgroup_events(
          count, reinterpret_cast<thread _simdgroup_event_t**>(events));
      }

    private:
      // Invoking the generation of LLVM bitcode for async copies.
      //
      //   %"struct.metal::simdgroup_event" = type { %struct._simdgroup_event_t* }
      //
      thread _simdgroup_event_t* event;
    };
  } // namespace metal
  #pragma METAL internals : disable

  #endif // __METAL_SIMDGROUP_EVENT
  """
}

/// Create the source code for the 'metal\_simdgroup\_matrix\_storage' header.
func createMetalSimdgroupMatrixStorage() -> String {
  // How this header spawning code was designed.
  //
  // Find the patterns between the load/store functions:
  // - device has 'uint' elements_per_row
  // - threadgroup has 'ushort' elements_per_row
  // - both have 'ushort2' matrix_origin
  //
  // The origin is 'ushort2' because the 32-bit part of the address should have
  // been applied previously during 'apply_offset'. The 16-bit part should be
  // hard-coded into the assembly when the GEMM loop is unrolled.
  //
  // Transpose path:
  // - load: reads two values; should split each one onto a separate line.
  //   - overwrites the value of *thread_elements() with a new vec<T, 2>
  // - store: the two instructions are on two separate lines.
  //   - fetches from lane 0 or 1 of thread_elements()[0]
  // - adds 0 or 1 to the hard-coded matrix_origin.x
  //
  // Address generation:
  // - casts some intermediate address fragments to 'ulong' for 'device'
  // - keeps all address fragments in 'ushort' for 'threadgroup'

  enum Action {
    case load
    case store
  }

  struct MemoryAccessDescriptor {
    var action: Action?
    var addressSpace: MTLAddressSpace?
    var decodingBF16: Bool?
    var indentationSpaceCount: Int = .zero
  }

  func createMemoryAccess(
    descriptor: MemoryAccessDescriptor
  )
    -> String
  {
    guard
      let action = descriptor.action,
      let addressSpace = descriptor.addressSpace,
      let decodingBF16 = descriptor.decodingBF16
    else {
      fatalError("Descriptor was incomplete.")
    }
    let indentation = String(
      repeating: " ", count: descriptor.indentationSpaceCount
    )

    // Determine the arguments.
    var arguments: [String] = []
    func addPointerArgument(dataType: String) {
      if action == .load {
        arguments.append("const \(addressSpace.keyword) \(dataType) *src")
      } else {
        arguments.append("\(addressSpace.keyword) \(dataType) *dst")
      }
    }
    if decodingBF16 {
      addPointerArgument(dataType: "bfloat")
    } else {
      addPointerArgument(dataType: "U")
    }
    arguments.append("\(addressSpace.offsetType) elements_per_row")
    arguments.append("ushort2 matrix_origin")
    arguments.append("bool transpose_matrix = false")

    // Create the warning comment.
    var output = ""
    if decodingBF16 {
      output += "\(indentation)// WARNING: 'T' must be 'float'.\n"
    } else {
      output += "\(indentation)template <typename U>\n"
    }

    // Create the function signature.
    output += "\(indentation)METAL_FUNC void"
    if action == .load {
      output += " load"
    } else {
      output += " store"
    }
    if decodingBF16 {
      output += "_bfloat"
    }
    output += "("
    for argumentID in arguments.indices {
      let argument = arguments[argumentID]
      output += argument
      if argumentID < arguments.count - 1 {
        output += ", "
      }
    }
    output += ") {\n"

    func createAddress(transposed: Bool, offset: Int) -> String {
      let lineY = "\(addressSpace.offsetType)(matrix_origin.y)"
      var lineX = "matrix_origin.x + \(offset)"
      lineX = "\(addressSpace.offsetType)(\(lineX))"

      if transposed {
        return "\(lineX) * elements_per_row + \(lineY)"
      } else {
        return "\(lineY) * elements_per_row + \(lineX)"
      }
    }

    func createTwoPartAccess(transposed: Bool) -> [String] {
      // Generate the addresses.
      var lines: [String] = []
      for laneID in 0..<2 {
        lines.append(
          "\(addressSpace.offsetType) address\(laneID) = "
            + createAddress(transposed: transposed, offset: laneID)
        )
      }

      if action == .load {
        if decodingBF16 {
          lines.append("bfloat memoryForm0 = src[address0]")
          lines.append("bfloat memoryForm1 = src[address1]")
        } else {
          lines.append("U memoryForm0 = src[address0]")
          lines.append("U memoryForm1 = src[address1]")
        }
      }

      if action == .load {
        if decodingBF16 {
          // Separate the loading logic from the decoding logic for clarity.
          lines.append(
            ""
          )

          // BF16 decoding logic.
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())"
          )
          lines.append(
            "registerForm[1] = memoryForm0"
          )
          lines.append(
            "registerForm[3] = memoryForm1"
          )
          lines.append(
            "((thread bfloat4*)thread_elements())[0] = registerForm"
          )
        } else {
          // Perform a type cast natively supported by the hardware.
          lines.append("((thread T*)thread_elements())[0] = T(memoryForm0)")
          lines.append("((thread T*)thread_elements())[1] = T(memoryForm1)")
        }
      } else {
        if decodingBF16 {
          // BF16 encoding logic.
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())"
          )
          lines.append(
            "registerForm[2] = registerForm[1]"
          )
        } else {
          // Type casts supported natively by the hardware.
          lines.append("T registerForm0 = ((thread T*)thread_elements())[0]")
          lines.append("T registerForm1 = ((thread T*)thread_elements())[1]")
        }
      }

      if action == .store {
        if decodingBF16 {
          lines.append("dst[address0] = registerForm[2]")
          lines.append("dst[address1] = registerForm[3]")
        } else {
          lines.append("dst[address0] = U(registerForm0)")
          lines.append("dst[address1] = U(registerForm1)")
        }
      }
      return lines
    }

    func createOnePartAccess() -> [String] {
      var lines: [String] = []
      do {
        let address = createAddress(transposed: false, offset: 0)
        lines.append("auto combinedAddress = \(address)")
      }
      if action == .load {
        if decodingBF16 {
          lines.append(
            "bfloat2 memoryForm = "
              + "*(const \(addressSpace.keyword) packed_bfloat2*)(src + combinedAddress)"
          )

          // Separate the loading logic from the decoding logic for clarity.
          lines.append(
            ""
          )

          // BF16 decoding logic.
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())"
          )
          lines.append(
            "((thread float*)&registerForm)[1] = *(thread float*)(&memoryForm)"
          )
          lines.append(
            "((thread bfloat*)&registerForm)[1] = memoryForm[0]"
          )
          lines.append(
            "((thread bfloat4*)thread_elements())[0] = registerForm"
          )
        } else {
          lines.append(
            "vec<U, 2> memoryForm = "
              + "*(const \(addressSpace.keyword) vec<U, 2>*)(src + combinedAddress)"
          )
          lines.append(
            "*(thread_elements()) = vec<T, 2>(memoryForm)"
          )
        }
      } else {
        if decodingBF16 {
          // BF16 encoding logic.
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())"
          )
          lines.append(
            "registerForm[2] = registerForm[1]"
          )
          lines.append(
            "float memoryForm = ((thread float*)&registerForm)[1]"
          )
          lines.append(
            "*(\(addressSpace.keyword) float*)(dst + combinedAddress) = " + "memoryForm"
          )
        } else {
          lines.append(
            "vec<T, 2> registerForm = *(thread_elements())"
          )
          lines.append(
            "*(\(addressSpace.keyword) vec<U, 2>*)(dst + combinedAddress) = "
              + "vec<U, 2>(registerForm)"
          )
        }
      }
      return lines
    }

    func addBlockContents(_ block: [String]) -> [String] {
      block.map {
        if $0.allSatisfy(\.isWhitespace) {
          "  "
        } else {
          "  \($0);"
        }
      }
    }

    // Determine the lines of the 'if' block.
    var body: [String] = []
    body.append("if (transpose_matrix) {")
    body += addBlockContents(createTwoPartAccess(transposed: true))

    // Determine the lines of the 'else' block.
    if decodingBF16 {
      let blockContents: [String] = if action == .load {
        createOnePartAccess()
      } else {
        createTwoPartAccess(transposed: false)
      }

      body.append("} else {")
      body += addBlockContents(blockContents)
      body.append("}")
    } else {
      body.append("} else if (elements_per_row % 2 != 0) {")
      body += addBlockContents(createTwoPartAccess(transposed: false))
      body.append("} else {")
      body += addBlockContents(createOnePartAccess())
      body.append("}")
    }

    // Create the function body.
    for line in body {
      output += "\(indentation)  \(line)\n"
    }
    output += "\(indentation)}\n"
    return output
  }

  // Add the first section of the shader.
  var output = ""
  output += """
  // -*- Metal -*-
  //===-- metal_simdgroup_matrix_storage ------------------------------------===//
  // Copyright (c) 2024 Philip Turner. See MIT LICENSE
  //===----------------------------------------------------------------------===//

  #ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
  #define __METAL_SIMDGROUP_MATRIX_STORAGE

  #include "GEMMBFloatTypes.h"

  // The layout of threads within a SIMD matrix.
  //
  //  0  0  1  1  8  8  9  9
  //  2  2  3  3 10 10 11 11
  //  4  4  5  5 12 12 13 13
  //  6  6  7  7 14 14 15 15
  // 16 16 17 17 24 24 25 25
  // 18 18 19 19 26 26 27 27
  // 20 20 21 21 28 28 29 29
  // 22 22 23 23 30 30 31 31
  //
  // This is Morton order, a method for coalescing data accesses. It is used
  // in a variety of contexts, from ray tracing acceleration structures, to
  // nodal-point Laplacians, to sorting large lattices of atoms.
  //
  // Source: https://patents.google.com/patent/US11256518B2
  METAL_FUNC static ushort2 morton_order(ushort thread_index_in_simdgroup) {
    ushort lane_id = thread_index_in_simdgroup;
    ushort quad_id = lane_id / 4;

    constexpr ushort QUADRANT_SPAN_M = 4;
    constexpr ushort THREADS_PER_QUADRANT = 8;
    ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
    ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
    ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;

    ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
    ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
    ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;

    return ushort2(N_in_simd, M_in_simd);
  }

  #pragma METAL internals : enable
  namespace metal
  {
    template <typename T>
    struct simdgroup_matrix_storage {
      typedef vec<T, 64> storage_type;

      storage_type t;

      METAL_FUNC thread vec<T, 2>* thread_elements() thread {
        return reinterpret_cast<thread vec<T, 2>*>(&t);
      }

      METAL_FUNC simdgroup_matrix_storage() thread = default;

      METAL_FUNC simdgroup_matrix_storage(vec<T, 2> thread_elements) thread {
        *(this->thread_elements()) = thread_elements;
      }

      METAL_FUNC explicit simdgroup_matrix_storage(T value) thread {
        *(this->thread_elements()) = vec<T, 2>(value);
      }

      METAL_FUNC static device T* apply_offset(device T *src, uint elements_per_row, uint2 matrix_origin, bool transpose_matrix = false) {
        if (transpose_matrix) {
          return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
        } else {
          return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
        }
      }

      METAL_FUNC static threadgroup T* apply_offset(threadgroup T *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
        if (transpose_matrix) {
          return src + matrix_origin.x * elements_per_row + matrix_origin.y;
        } else {
          return src + matrix_origin.y * elements_per_row + matrix_origin.x;
        }
      }

  """

  var desc = MemoryAccessDescriptor()
  desc.indentationSpaceCount = 4

  for action in [Action.load, .store] {
    for addressSpace in [MTLAddressSpace.device, .threadgroup] {
      for decodingBF16 in [false, true] {
        desc.action = action
        desc.addressSpace = addressSpace

        desc.decodingBF16 = decodingBF16
        output += createMemoryAccess(descriptor: desc)
        output += "\n"
      }
    }
  }

  // Add quantized load methods
  output += """
      // Quantized INT8 load method - device memory
      METAL_FUNC void load_quantized_int8(const device char *src, uint elements_per_row, ushort2 matrix_origin, float scale, int32_t zero_point, bool transpose_matrix = false) {
        static_assert(is_same_v<T, float>, "Quantized load only supported for float register precision");

        // Calculate address manually for int8_t data
        const device char *adjusted_src;
        if (transpose_matrix) {
          adjusted_src = src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
        } else {
          adjusted_src = src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
        }

        if (!transpose_matrix) {
          if (elements_per_row % 2 == 0) {
            vec<int8_t, 2> quantized_data = *reinterpret_cast<const device vec<int8_t, 2>*>(adjusted_src);
            vec<T, 2> dequantized_data;
            dequantized_data.x = (float(quantized_data.x) - float(zero_point)) * scale;
            dequantized_data.y = (float(quantized_data.y) - float(zero_point)) * scale;
            *(thread_elements()) = dequantized_data;
          } else {
            T val1 = (float(adjusted_src[0]) - float(zero_point)) * scale;
            T val2 = (elements_per_row > 1) ? (float(adjusted_src[1]) - float(zero_point)) * scale : T(0);
            *(thread_elements()) = vec<T, 2>(val1, val2);
          }
        } else {
          T val1 = (float(adjusted_src[0]) - float(zero_point)) * scale;
          T val2 = (elements_per_row > 1) ? (float(adjusted_src[elements_per_row]) - float(zero_point)) * scale : T(0);
          *(thread_elements()) = vec<T, 2>(val1, val2);
        }
      }

      // Quantized INT8 load method - threadgroup memory
      METAL_FUNC void load_quantized_int8(const threadgroup char *src, ushort elements_per_row, ushort2 matrix_origin, float scale, int32_t zero_point, bool transpose_matrix = false) {
        static_assert(is_same_v<T, float>, "Quantized load only supported for float register precision");

        // Calculate address manually for int8_t data
        const threadgroup char *adjusted_src;
        if (transpose_matrix) {
          adjusted_src = src + matrix_origin.x * elements_per_row + matrix_origin.y;
        } else {
          adjusted_src = src + matrix_origin.y * elements_per_row + matrix_origin.x;
        }

        if (!transpose_matrix) {
          if (elements_per_row % 2 == 0) {
            vec<int8_t, 2> quantized_data = *reinterpret_cast<const threadgroup vec<int8_t, 2>*>(adjusted_src);
            vec<T, 2> dequantized_data;
            dequantized_data.x = (float(quantized_data.x) - float(zero_point)) * scale;
            dequantized_data.y = (float(quantized_data.y) - float(zero_point)) * scale;
            *(thread_elements()) = dequantized_data;
          } else {
            T val1 = (float(adjusted_src[0]) - float(zero_point)) * scale;
            T val2 = (elements_per_row > 1) ? (float(adjusted_src[1]) - float(zero_point)) * scale : T(0);
            *(thread_elements()) = vec<T, 2>(val1, val2);
          }
        } else {
          T val1 = (float(adjusted_src[0]) - float(zero_point)) * scale;
          T val2 = (elements_per_row > 1) ? (float(adjusted_src[elements_per_row]) - float(zero_point)) * scale : T(0);
          *(thread_elements()) = vec<T, 2>(val1, val2);
        }
      }

      // Quantized INT4 load method - device memory
      METAL_FUNC void load_quantized_int4(const device uchar *src, uint elements_per_row, ushort2 matrix_origin, float scale, int32_t zero_point, bool transpose_matrix = false) {
        static_assert(is_same_v<T, float>, "Quantized load only supported for float register precision");

        // For INT4, elements are packed 2 per byte
        uint packed_elements_per_row = (elements_per_row + 1) / 2;

        // Calculate address manually for packed INT4 data
        const device uchar *adjusted_src;
        if (transpose_matrix) {
          // For transpose: swap x/y coordinates in addressing
          adjusted_src = src + ulong((matrix_origin.x / 2) * packed_elements_per_row) + (matrix_origin.y / 2);
        } else {
          // Normal addressing: row-major order
          adjusted_src = src + ulong((matrix_origin.y / 2) * packed_elements_per_row) + (matrix_origin.x / 2);
        }

        uint8_t packed = adjusted_src[0];
        int32_t val1, val2;

        if (matrix_origin.x % 2 == 0) {
          val1 = int32_t(packed & 0xF) - 8;  // Lower nibble, convert from [0,15] to [-8,7]
          val2 = int32_t(packed >> 4) - 8;   // Upper nibble
        } else {
          val1 = int32_t(packed >> 4) - 8;   // Upper nibble from current byte
          // val2 should come from lower nibble of next byte
          val2 = (matrix_origin.x + 1 < elements_per_row) ?
                 int32_t((adjusted_src[1] & 0xF)) - 8 : 0;
        }

        T dequant_val1 = (float(val1) - float(zero_point)) * scale;
        T dequant_val2 = (float(val2) - float(zero_point)) * scale;
        *(thread_elements()) = vec<T, 2>(dequant_val1, dequant_val2);
      }

      // Quantized INT4 load method - threadgroup memory
      METAL_FUNC void load_quantized_int4(const threadgroup uchar *src, ushort elements_per_row, ushort2 matrix_origin, float scale, int32_t zero_point, bool transpose_matrix = false) {
        static_assert(is_same_v<T, float>, "Quantized load only supported for float register precision");

        // For INT4, elements are packed 2 per byte
        ushort packed_elements_per_row = (elements_per_row + 1) / 2;

        // Calculate address manually for packed INT4 data
        const threadgroup uchar *adjusted_src;
        if (transpose_matrix) {
          // For transpose: swap x/y coordinates in addressing
          adjusted_src = src + (matrix_origin.x / 2) * packed_elements_per_row + (matrix_origin.y / 2);
        } else {
          // Normal addressing: row-major order
          adjusted_src = src + (matrix_origin.y / 2) * packed_elements_per_row + (matrix_origin.x / 2);
        }

        uint8_t packed = adjusted_src[0];
        int32_t val1, val2;

        if (matrix_origin.x % 2 == 0) {
          val1 = int32_t(packed & 0xF) - 8;  // Lower nibble, convert from [0,15] to [-8,7]
          val2 = int32_t(packed >> 4) - 8;   // Upper nibble
        } else {
          val1 = int32_t(packed >> 4) - 8;   // Upper nibble from current byte
          // val2 should come from lower nibble of next byte
          val2 = (matrix_origin.x + 1 < elements_per_row) ?
                 int32_t((adjusted_src[1] & 0xF)) - 8 : 0;
        }

        T dequant_val1 = (float(val1) - float(zero_point)) * scale;
        T dequant_val2 = (float(val2) - float(zero_point)) * scale;
        *(thread_elements()) = vec<T, 2>(dequant_val1, dequant_val2);
      }

  """

  // Add the last section of the header.
  output += """
      template <typename U, typename V>
      METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) {
        if (!accumulate) {
          *(thread_elements()) = vec<T, 2>(0);
        }
        t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
      }
    };
  } // namespace metal
  #pragma METAL internals : disable

  #endif // __METAL_SIMDGROUP_MATRIX_STORAGE

  """
  return output
}

/// Create the source code for runtime quantization kernel headers.
func createRuntimeQuantizationHeaders() -> String {
  """
  // -*- Metal -*-
  //===-- runtime_quantization_headers -------------------------------------===//
  // Copyright (c) 2024 Philip Turner. See MIT LICENSE
  //===----------------------------------------------------------------------===//

  #ifndef __RUNTIME_QUANTIZATION_HEADERS
  #define __RUNTIME_QUANTIZATION_HEADERS

  // Forward declarations for runtime quantization kernels

  // ============================================================================
  // TENSOR-WISE QUANTIZATION KERNEL DECLARATIONS
  // ============================================================================

  // FP32 → INT8 tensor quantization
  kernel void quantize_tensor_fp32_to_int8(
      device float *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      constant uint &count [[buffer(2)]],
      constant float &scale [[buffer(3)]],
      constant int &zero_point [[buffer(4)]],
      uint gid [[thread_position_in_grid]]
  );

  // FP16 → INT8 tensor quantization
  kernel void quantize_tensor_fp16_to_int8(
      device half *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      constant uint &count [[buffer(2)]],
      constant float &scale [[buffer(3)]],
      constant int &zero_point [[buffer(4)]],
      uint gid [[thread_position_in_grid]]
  );

  // BF16 → INT8 tensor quantization
  kernel void quantize_tensor_bf16_to_int8(
      device bfloat *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      constant uint &count [[buffer(2)]],
      constant float &scale [[buffer(3)]],
      constant int &zero_point [[buffer(4)]],
      uint gid [[thread_position_in_grid]]
  );

  // FP32 → INT4 tensor quantization
  kernel void quantize_tensor_fp32_to_int4(
      device float *input [[buffer(0)]],
      device uchar *output [[buffer(1)]],
      constant uint &count [[buffer(2)]],
      constant float &scale [[buffer(3)]],
      constant int &zero_point [[buffer(4)]],
      uint gid [[thread_position_in_grid]]
  );

  // FP16 → INT4 tensor quantization
  kernel void quantize_tensor_fp16_to_int4(
      device half *input [[buffer(0)]],
      device uchar *output [[buffer(1)]],
      constant uint &count [[buffer(2)]],
      constant float &scale [[buffer(3)]],
      constant int &zero_point [[buffer(4)]],
      uint gid [[thread_position_in_grid]]
  );

  // BF16 → INT4 tensor quantization
  kernel void quantize_tensor_bf16_to_int4(
      device bfloat *input [[buffer(0)]],
      device uchar *output [[buffer(1)]],
      constant uint &count [[buffer(2)]],
      constant float &scale [[buffer(3)]],
      constant int &zero_point [[buffer(4)]],
      uint gid [[thread_position_in_grid]]
  );

  // ============================================================================
  // BLOCK-WISE QUANTIZATION KERNEL DECLARATIONS
  // ============================================================================

  // FP32 → INT8 block-wise quantization
  kernel void quantize_blockwise_fp32_to_int8(
      device float *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      device float *scales [[buffer(2)]],
      constant uint &rows [[buffer(3)]],
      constant uint &cols [[buffer(4)]],
      constant uint &block_size [[buffer(5)]],
      uint3 gid [[threadgroup_position_in_grid]],
      uint tid [[thread_position_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  );

  // FP16 → INT8 block-wise quantization
  kernel void quantize_blockwise_fp16_to_int8(
      device half *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      device float *scales [[buffer(2)]],
      constant uint &rows [[buffer(3)]],
      constant uint &cols [[buffer(4)]],
      constant uint &block_size [[buffer(5)]],
      uint3 gid [[threadgroup_position_in_grid]],
      uint tid [[thread_position_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  );

  // BF16 → INT8 block-wise quantization
  kernel void quantize_blockwise_bf16_to_int8(
      device bfloat *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      device float *scales [[buffer(2)]],
      constant uint &rows [[buffer(3)]],
      constant uint &cols [[buffer(4)]],
      constant uint &block_size [[buffer(5)]],
      uint3 gid [[threadgroup_position_in_grid]],
      uint tid [[thread_position_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  );

  // ============================================================================
  // ROW-WISE QUANTIZATION KERNEL DECLARATIONS
  // ============================================================================

  // FP32 → INT8 row-wise quantization
  kernel void quantize_rowwise_fp32_to_int8(
      device float *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      device float *scales [[buffer(2)]],
      constant uint &rows [[buffer(3)]],
      constant uint &cols [[buffer(4)]],
      uint gid [[threadgroup_position_in_grid]],
      uint tid [[thread_position_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  );

  // FP16 → INT8 row-wise quantization
  kernel void quantize_rowwise_fp16_to_int8(
      device half *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      device float *scales [[buffer(2)]],
      constant uint &rows [[buffer(3)]],
      constant uint &cols [[buffer(4)]],
      uint gid [[threadgroup_position_in_grid]],
      uint tid [[thread_position_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  );

  // BF16 → INT8 row-wise quantization
  kernel void quantize_rowwise_bf16_to_int8(
      device bfloat *input [[buffer(0)]],
      device char *output [[buffer(1)]],
      device float *scales [[buffer(2)]],
      constant uint &rows [[buffer(3)]],
      constant uint &cols [[buffer(4)]],
      uint gid [[threadgroup_position_in_grid]],
      uint tid [[thread_position_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  );

  #endif // __RUNTIME_QUANTIZATION_HEADERS
  """
}

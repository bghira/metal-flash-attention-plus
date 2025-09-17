//
//  GEMMRuntimeQuantization.metal
//  FlashAttention
//
//  Runtime quantization kernels for fp16/bf16/fp32 → int8/int4 conversion
//  Optimized with vectorized operations and simdgroup reductions
//

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

namespace runtime_quantization {

    // Efficient simdgroup min/max reduction for scale computation
    METAL_FUNC float simdgroup_min_reduce(float value) {
        float result = value;
        for (ushort lane_offset = 16; lane_offset >= 1; lane_offset >>= 1) {
            result = min(result, simd_shuffle_down(result, lane_offset));
        }
        return simd_broadcast_first(result);
    }

    METAL_FUNC float simdgroup_max_reduce(float value) {
        float result = value;
        for (ushort lane_offset = 16; lane_offset >= 1; lane_offset >>= 1) {
            result = max(result, simd_shuffle_down(result, lane_offset));
        }
        return simd_broadcast_first(result);
    }

    // Vectorized FP32 → INT8 quantization with float4 operations
    METAL_FUNC char4 quantize_fp32_to_int8_vec4(float4 input, float scale, int zero_point) {
        int4 quantized = int4(round(input / scale)) + zero_point;
        return char4(clamp(quantized, -128, 127));
    }

    // Vectorized FP16 → INT8 quantization
    METAL_FUNC char4 quantize_fp16_to_int8_vec4(half4 input, float scale, int zero_point) {
        float4 fp32_input = float4(input);
        return quantize_fp32_to_int8_vec4(fp32_input, scale, zero_point);
    }

    // Vectorized BF16 → INT8 quantization
    METAL_FUNC char4 quantize_bf16_to_int8_vec4(packed_bfloat4 input, float scale, int zero_point) {
        float4 fp32_input = float4(input);
        return quantize_fp32_to_int8_vec4(fp32_input, scale, zero_point);
    }

    // Vectorized FP32 → INT4 quantization (2 values per byte)
    METAL_FUNC uchar2 quantize_fp32_to_int4_vec4(float4 input, float scale, int zero_point) {
        int4 quantized = int4(round(input / scale)) + zero_point;
        int4 clamped = clamp(quantized + 8, 0, 15); // Convert [-8,7] to [0,15]

        uchar val1 = uchar(clamped.x) | (uchar(clamped.y) << 4);
        uchar val2 = uchar(clamped.z) | (uchar(clamped.w) << 4);
        return uchar2(val1, val2);
    }

    // Vectorized FP16 → INT4 quantization
    METAL_FUNC uchar2 quantize_fp16_to_int4_vec4(half4 input, float scale, int zero_point) {
        float4 fp32_input = float4(input);
        return quantize_fp32_to_int4_vec4(fp32_input, scale, zero_point);
    }

    // Vectorized BF16 → INT4 quantization
    METAL_FUNC uchar2 quantize_bf16_to_int4_vec4(packed_bfloat4 input, float scale, int zero_point) {
        float4 fp32_input = float4(input);
        return quantize_fp32_to_int4_vec4(fp32_input, scale, zero_point);
    }

    // Block-wise scale computation with simdgroup optimization
    template<typename T>
    METAL_FUNC float compute_block_scale_simdgroup(
        const device T *data,
        uint block_start,
        uint block_size,
        uint stride,
        uint rows_in_block,
        uint cols_in_block,
        ushort thread_id_in_simdgroup
    ) {
        float local_min = INFINITY;
        float local_max = -INFINITY;

        // Each thread processes multiple elements for coalesced access
        for (uint row = 0; row < rows_in_block; row++) {
            for (uint col = thread_id_in_simdgroup; col < cols_in_block; col += 32) {
                uint idx = block_start + row * stride + col;
                float val = float(data[idx]);
                local_min = min(local_min, val);
                local_max = max(local_max, val);
            }
        }

        // Reduce across simdgroup
        float block_min = simdgroup_min_reduce(local_min);
        float block_max = simdgroup_max_reduce(local_max);

        // Symmetric quantization scale
        float abs_max = max(abs(block_min), abs(block_max));
        return abs_max / 127.0f; // For INT8
    }

    // Row-wise scale computation with simdgroup optimization
    template<typename T>
    METAL_FUNC float compute_row_scale_simdgroup(
        const device T *data,
        uint row_start_idx,
        uint row_length,
        ushort thread_id_in_simdgroup
    ) {
        float local_min = INFINITY;
        float local_max = -INFINITY;

        // Process row elements with stride 32 for simdgroup efficiency
        for (uint col = thread_id_in_simdgroup; col < row_length; col += 32) {
            float val = float(data[row_start_idx + col]);
            local_min = min(local_min, val);
            local_max = max(local_max, val);
        }

        // Reduce across simdgroup
        float row_min = simdgroup_min_reduce(local_min);
        float row_max = simdgroup_max_reduce(local_max);

        float abs_max = max(abs(row_min), abs(row_max));
        return abs_max / 127.0f; // For INT8
    }
}

// ============================================================================
// TENSOR-WISE QUANTIZATION KERNELS
// ============================================================================

kernel void quantize_tensor_fp32_to_int8(
    device float *input [[buffer(0)]],
    device char *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant int &zero_point [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= count) return;

    float4 input_vec = {0, 0, 0, 0};
    // Vectorized load with bounds checking
    if (idx < count) input_vec.x = input[idx];
    if (idx + 1 < count) input_vec.y = input[idx + 1];
    if (idx + 2 < count) input_vec.z = input[idx + 2];
    if (idx + 3 < count) input_vec.w = input[idx + 3];

    char4 quantized = runtime_quantization::quantize_fp32_to_int8_vec4(input_vec, scale, zero_point);

    // Vectorized store with bounds checking
    if (idx < count) output[idx] = quantized.x;
    if (idx + 1 < count) output[idx + 1] = quantized.y;
    if (idx + 2 < count) output[idx + 2] = quantized.z;
    if (idx + 3 < count) output[idx + 3] = quantized.w;
}

kernel void quantize_tensor_fp16_to_int8(
    device half *input [[buffer(0)]],
    device char *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant int &zero_point [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= count) return;

    half4 input_vec = {0, 0, 0, 0};
    if (idx < count) input_vec.x = input[idx];
    if (idx + 1 < count) input_vec.y = input[idx + 1];
    if (idx + 2 < count) input_vec.z = input[idx + 2];
    if (idx + 3 < count) input_vec.w = input[idx + 3];

    char4 quantized = runtime_quantization::quantize_fp16_to_int8_vec4(input_vec, scale, zero_point);

    if (idx < count) output[idx] = quantized.x;
    if (idx + 1 < count) output[idx + 1] = quantized.y;
    if (idx + 2 < count) output[idx + 2] = quantized.z;
    if (idx + 3 < count) output[idx + 3] = quantized.w;
}

kernel void quantize_tensor_bf16_to_int8(
    device bfloat *input [[buffer(0)]],
    device char *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant int &zero_point [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= count) return;

    packed_bfloat4 input_vec;
    input_vec.x = (idx < count) ? input[idx] : bfloat(0);
    input_vec.y = (idx + 1 < count) ? input[idx + 1] : bfloat(0);
    input_vec.z = (idx + 2 < count) ? input[idx + 2] : bfloat(0);
    input_vec.w = (idx + 3 < count) ? input[idx + 3] : bfloat(0);

    char4 quantized = runtime_quantization::quantize_bf16_to_int8_vec4(input_vec, scale, zero_point);

    if (idx < count) output[idx] = quantized.x;
    if (idx + 1 < count) output[idx + 1] = quantized.y;
    if (idx + 2 < count) output[idx + 2] = quantized.z;
    if (idx + 3 < count) output[idx + 3] = quantized.w;
}

// INT4 tensor quantization kernels
kernel void quantize_tensor_fp32_to_int4(
    device float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant int &zero_point [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= count) return;

    float4 input_vec = {0, 0, 0, 0};
    if (idx < count) input_vec.x = input[idx];
    if (idx + 1 < count) input_vec.y = input[idx + 1];
    if (idx + 2 < count) input_vec.z = input[idx + 2];
    if (idx + 3 < count) input_vec.w = input[idx + 3];

    uchar2 quantized = runtime_quantization::quantize_fp32_to_int4_vec4(input_vec, scale, zero_point);

    // Store packed INT4 values
    uint output_idx = gid * 2;
    if (output_idx < (count + 1) / 2) output[output_idx] = quantized.x;
    if (output_idx + 1 < (count + 1) / 2) output[output_idx + 1] = quantized.y;
}

kernel void quantize_tensor_fp16_to_int4(
    device half *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant int &zero_point [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= count) return;

    half4 input_vec = {0, 0, 0, 0};
    if (idx < count) input_vec.x = input[idx];
    if (idx + 1 < count) input_vec.y = input[idx + 1];
    if (idx + 2 < count) input_vec.z = input[idx + 2];
    if (idx + 3 < count) input_vec.w = input[idx + 3];

    uchar2 quantized = runtime_quantization::quantize_fp16_to_int4_vec4(input_vec, scale, zero_point);

    uint output_idx = gid * 2;
    if (output_idx < (count + 1) / 2) output[output_idx] = quantized.x;
    if (output_idx + 1 < (count + 1) / 2) output[output_idx + 1] = quantized.y;
}

kernel void quantize_tensor_bf16_to_int4(
    device bfloat *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    constant int &zero_point [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid * 4;
    if (idx >= count) return;

    packed_bfloat4 input_vec;
    input_vec.x = (idx < count) ? input[idx] : bfloat(0);
    input_vec.y = (idx + 1 < count) ? input[idx + 1] : bfloat(0);
    input_vec.z = (idx + 2 < count) ? input[idx + 2] : bfloat(0);
    input_vec.w = (idx + 3 < count) ? input[idx + 3] : bfloat(0);

    uchar2 quantized = runtime_quantization::quantize_bf16_to_int4_vec4(input_vec, scale, zero_point);

    uint output_idx = gid * 2;
    if (output_idx < (count + 1) / 2) output[output_idx] = quantized.x;
    if (output_idx + 1 < (count + 1) / 2) output[output_idx + 1] = quantized.y;
}

// ============================================================================
// BLOCK-WISE QUANTIZATION KERNELS
// ============================================================================

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
) {
    // Block coordinates
    uint block_row = gid.y;
    uint block_col = gid.x;
    uint blocks_per_row = (cols + block_size - 1) / block_size;
    uint blocks_per_col = (rows + block_size - 1) / block_size;

    if (block_row >= blocks_per_col || block_col >= blocks_per_row) return;

    uint block_start_row = block_row * block_size;
    uint block_start_col = block_col * block_size;
    uint block_rows = min(block_size, rows - block_start_row);
    uint block_cols = min(block_size, cols - block_start_col);

    uint block_start_idx = block_start_row * cols + block_start_col;
    uint scale_idx = block_row * blocks_per_row + block_col;

    // Compute scale for this block using simdgroup reduction
    float scale = runtime_quantization::compute_block_scale_simdgroup(
        input, block_start_idx, block_size, cols, block_rows, block_cols, simd_lane_id);

    // Store scale (only first thread in simdgroup)
    if (simd_lane_id == 0) {
        scales[scale_idx] = scale;
    }

    // Quantize block data with vectorized operations
    for (uint row = 0; row < block_rows; row++) {
        for (uint col = tid * 4; col < block_cols; col += 32 * 4) { // 32 threads * 4 elements
            uint base_idx = block_start_idx + row * cols + col;

            if (col + 3 < block_cols) {
                float4 input_vec = {input[base_idx], input[base_idx + 1],
                                   input[base_idx + 2], input[base_idx + 3]};
                char4 quantized = runtime_quantization::quantize_fp32_to_int8_vec4(input_vec, scale, 0);

                output[base_idx] = quantized.x;
                output[base_idx + 1] = quantized.y;
                output[base_idx + 2] = quantized.z;
                output[base_idx + 3] = quantized.w;
            } else {
                // Handle remainder elements
                for (uint i = 0; i < 4 && col + i < block_cols; i++) {
                    float val = input[base_idx + i];
                    int quantized = int(round(val / scale));
                    output[base_idx + i] = char(clamp(quantized, -128, 127));
                }
            }
        }
    }
}

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
) {
    uint block_row = gid.y;
    uint block_col = gid.x;
    uint blocks_per_row = (cols + block_size - 1) / block_size;
    uint blocks_per_col = (rows + block_size - 1) / block_size;

    if (block_row >= blocks_per_col || block_col >= blocks_per_row) return;

    uint block_start_row = block_row * block_size;
    uint block_start_col = block_col * block_size;
    uint block_rows = min(block_size, rows - block_start_row);
    uint block_cols = min(block_size, cols - block_start_col);

    uint block_start_idx = block_start_row * cols + block_start_col;
    uint scale_idx = block_row * blocks_per_row + block_col;

    float scale = runtime_quantization::compute_block_scale_simdgroup(
        input, block_start_idx, block_size, cols, block_rows, block_cols, simd_lane_id);

    if (simd_lane_id == 0) {
        scales[scale_idx] = scale;
    }

    for (uint row = 0; row < block_rows; row++) {
        for (uint col = tid * 4; col < block_cols; col += 32 * 4) {
            uint base_idx = block_start_idx + row * cols + col;

            if (col + 3 < block_cols) {
                half4 input_vec = {input[base_idx], input[base_idx + 1],
                                  input[base_idx + 2], input[base_idx + 3]};
                char4 quantized = runtime_quantization::quantize_fp16_to_int8_vec4(input_vec, scale, 0);

                output[base_idx] = quantized.x;
                output[base_idx + 1] = quantized.y;
                output[base_idx + 2] = quantized.z;
                output[base_idx + 3] = quantized.w;
            } else {
                for (uint i = 0; i < 4 && col + i < block_cols; i++) {
                    float val = float(input[base_idx + i]);
                    int quantized = int(round(val / scale));
                    output[base_idx + i] = char(clamp(quantized, -128, 127));
                }
            }
        }
    }
}

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
) {
    uint block_row = gid.y;
    uint block_col = gid.x;
    uint blocks_per_row = (cols + block_size - 1) / block_size;
    uint blocks_per_col = (rows + block_size - 1) / block_size;

    if (block_row >= blocks_per_col || block_col >= blocks_per_row) return;

    uint block_start_row = block_row * block_size;
    uint block_start_col = block_col * block_size;
    uint block_rows = min(block_size, rows - block_start_row);
    uint block_cols = min(block_size, cols - block_start_col);

    uint block_start_idx = block_start_row * cols + block_start_col;
    uint scale_idx = block_row * blocks_per_row + block_col;

    float scale = runtime_quantization::compute_block_scale_simdgroup(
        input, block_start_idx, block_size, cols, block_rows, block_cols, simd_lane_id);

    if (simd_lane_id == 0) {
        scales[scale_idx] = scale;
    }

    for (uint row = 0; row < block_rows; row++) {
        for (uint col = tid * 4; col < block_cols; col += 32 * 4) {
            uint base_idx = block_start_idx + row * cols + col;

            if (col + 3 < block_cols) {
                packed_bfloat4 input_vec;
                input_vec.x = input[base_idx];
                input_vec.y = input[base_idx + 1];
                input_vec.z = input[base_idx + 2];
                input_vec.w = input[base_idx + 3];

                char4 quantized = runtime_quantization::quantize_bf16_to_int8_vec4(input_vec, scale, 0);

                output[base_idx] = quantized.x;
                output[base_idx + 1] = quantized.y;
                output[base_idx + 2] = quantized.z;
                output[base_idx + 3] = quantized.w;
            } else {
                for (uint i = 0; i < 4 && col + i < block_cols; i++) {
                    float val = float(input[base_idx + i]);
                    int quantized = int(round(val / scale));
                    output[base_idx + i] = char(clamp(quantized, -128, 127));
                }
            }
        }
    }
}

// ============================================================================
// ROW-WISE QUANTIZATION KERNELS
// ============================================================================

kernel void quantize_rowwise_fp32_to_int8(
    device float *input [[buffer(0)]],
    device char *output [[buffer(1)]],
    device float *scales [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint row = gid;
    if (row >= rows) return;

    uint row_start_idx = row * cols;

    // Compute scale for this row using simdgroup reduction
    float scale = runtime_quantization::compute_row_scale_simdgroup(
        input, row_start_idx, cols, simd_lane_id);

    // Store scale (only first thread in simdgroup)
    if (simd_lane_id == 0) {
        scales[row] = scale;
    }

    // Quantize row data with vectorized operations
    for (uint col = tid * 4; col < cols; col += 32 * 4) { // 32 threads * 4 elements
        uint base_idx = row_start_idx + col;

        if (col + 3 < cols) {
            float4 input_vec = {input[base_idx], input[base_idx + 1],
                               input[base_idx + 2], input[base_idx + 3]};
            char4 quantized = runtime_quantization::quantize_fp32_to_int8_vec4(input_vec, scale, 0);

            output[base_idx] = quantized.x;
            output[base_idx + 1] = quantized.y;
            output[base_idx + 2] = quantized.z;
            output[base_idx + 3] = quantized.w;
        } else {
            // Handle remainder elements
            for (uint i = 0; i < 4 && col + i < cols; i++) {
                float val = input[base_idx + i];
                int quantized = int(round(val / scale));
                output[base_idx + i] = char(clamp(quantized, -128, 127));
            }
        }
    }
}

kernel void quantize_rowwise_fp16_to_int8(
    device half *input [[buffer(0)]],
    device char *output [[buffer(1)]],
    device float *scales [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint row = gid;
    if (row >= rows) return;

    uint row_start_idx = row * cols;

    float scale = runtime_quantization::compute_row_scale_simdgroup(
        input, row_start_idx, cols, simd_lane_id);

    if (simd_lane_id == 0) {
        scales[row] = scale;
    }

    for (uint col = tid * 4; col < cols; col += 32 * 4) {
        uint base_idx = row_start_idx + col;

        if (col + 3 < cols) {
            half4 input_vec = {input[base_idx], input[base_idx + 1],
                              input[base_idx + 2], input[base_idx + 3]};
            char4 quantized = runtime_quantization::quantize_fp16_to_int8_vec4(input_vec, scale, 0);

            output[base_idx] = quantized.x;
            output[base_idx + 1] = quantized.y;
            output[base_idx + 2] = quantized.z;
            output[base_idx + 3] = quantized.w;
        } else {
            for (uint i = 0; i < 4 && col + i < cols; i++) {
                float val = float(input[base_idx + i]);
                int quantized = int(round(val / scale));
                output[base_idx + i] = char(clamp(quantized, -128, 127));
            }
        }
    }
}

kernel void quantize_rowwise_bf16_to_int8(
    device bfloat *input [[buffer(0)]],
    device char *output [[buffer(1)]],
    device float *scales [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint row = gid;
    if (row >= rows) return;

    uint row_start_idx = row * cols;

    float scale = runtime_quantization::compute_row_scale_simdgroup(
        input, row_start_idx, cols, simd_lane_id);

    if (simd_lane_id == 0) {
        scales[row] = scale;
    }

    for (uint col = tid * 4; col < cols; col += 32 * 4) {
        uint base_idx = row_start_idx + col;

        if (col + 3 < cols) {
            packed_bfloat4 input_vec;
            input_vec.x = input[base_idx];
            input_vec.y = input[base_idx + 1];
            input_vec.z = input[base_idx + 2];
            input_vec.w = input[base_idx + 3];

            char4 quantized = runtime_quantization::quantize_bf16_to_int8_vec4(input_vec, scale, 0);

            output[base_idx] = quantized.x;
            output[base_idx + 1] = quantized.y;
            output[base_idx + 2] = quantized.z;
            output[base_idx + 3] = quantized.w;
        } else {
            for (uint i = 0; i < 4 && col + i < cols; i++) {
                float val = float(input[base_idx + i]);
                int quantized = int(round(val / scale));
                output[base_idx + i] = char(clamp(quantized, -128, 127));
            }
        }
    }
}
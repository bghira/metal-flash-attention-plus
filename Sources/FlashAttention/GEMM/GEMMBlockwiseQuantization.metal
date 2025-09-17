//
//  GEMMBlockwiseQuantization.metal
//  FlashAttention
//
//  Fused blockwise quantization kernels that compute block statistics
//  and quantize in a single pass for optimal memory bandwidth usage.
//

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

namespace blockwise_quantization {

    // Helper for simdgroup mean computation
    METAL_FUNC float compute_block_mean_simdgroup(
        device float* data,
        uint block_start_idx,
        uint block_size,
        uint total_elements,
        ushort simd_lane_id
    ) {
        float sum = 0.0f;
        uint count = 0;

        // Each thread accumulates its portion
        for (uint i = simd_lane_id; i < block_size && (block_start_idx + i) < total_elements; i += 32) {
            sum += data[block_start_idx + i];
            count++;
        }

        // Simdgroup reduction for sum and count
        sum = simd_sum(sum);
        count = simd_sum(count);

        return (count > 0) ? (sum / float(count)) : 0.0f;
    }

    // Helper for simdgroup min/max reduction after centering
    METAL_FUNC float2 compute_centered_minmax_simdgroup(
        device float* data,
        uint block_start_idx,
        uint block_size,
        uint total_elements,
        float block_mean,
        ushort simd_lane_id
    ) {
        float local_min = INFINITY;
        float local_max = -INFINITY;

        // Each thread finds local min/max of centered values
        for (uint i = simd_lane_id; i < block_size && (block_start_idx + i) < total_elements; i += 32) {
            float centered_val = data[block_start_idx + i] - block_mean;
            local_min = min(local_min, centered_val);
            local_max = max(local_max, centered_val);
        }

        // Simdgroup reduction
        float block_min = simd_min(local_min);
        float block_max = simd_max(local_max);

        return float2(block_min, block_max);
    }

    // BF16 to FP16 conversion helper
    METAL_FUNC half bf16_to_fp16(ushort bf16_bits) {
        // Expand BF16 to FP32 then convert to FP16
        uint fp32_bits = uint(bf16_bits) << 16;
        float fp32_val = as_type<float>(fp32_bits);
        return half(fp32_val);
    }

    // Vectorized quantization helper for FP32
    METAL_FUNC char4 quantize_fp32_vec4_asymmetric(float4 input, float scale, int8_t zero_point) {
        int4 quantized = int4(round(input / scale)) + int4(zero_point);
        return char4(clamp(quantized, -128, 127));
    }

    // Vectorized quantization helper for FP16
    METAL_FUNC char4 quantize_fp16_vec4_asymmetric(half4 input, float scale, int8_t zero_point) {
        float4 fp32_input = float4(input);
        return quantize_fp32_vec4_asymmetric(fp32_input, scale, zero_point);
    }

    // Vectorized quantization helper for BF16
    METAL_FUNC char4 quantize_bf16_vec4_asymmetric(ushort4 bf16_input, float scale, int8_t zero_point) {
        float4 fp32_input = float4(
            bf16_to_fp16(bf16_input.x),
            bf16_to_fp16(bf16_input.y),
            bf16_to_fp16(bf16_input.z),
            bf16_to_fp16(bf16_input.w)
        );
        return quantize_fp32_vec4_asymmetric(fp32_input, scale, zero_point);
    }
}

// ============================================================================
// FUSED BLOCKWISE CENTERED QUANTIZATION KERNELS
// ============================================================================

kernel void quantize_blockwise_centered_fp32_to_int8(
    device float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device float* blockScales [[buffer(2)]],
    device int8_t* blockZeroPoints [[buffer(3)]],
    device int32_t* blockSums [[buffer(4)]], // Optional: precompute Σ Q
    constant uint& K [[buffer(5)]],
    constant uint& blockSizeK [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]
) {
    const uint block_idx = gid / blockSizeK;
    const uint num_blocks = (K + blockSizeK - 1) / blockSizeK;

    if (block_idx >= num_blocks) return;

    const uint block_start = block_idx * blockSizeK;
    const uint block_end = min(block_start + blockSizeK, K);
    const uint block_elements = block_end - block_start;

    // Step 1: Compute block mean using simdgroup operations
    float block_mean = blockwise_quantization::compute_block_mean_simdgroup(
        input, block_start, block_elements, K, simd_lane_id
    );

    // Step 2: Find min/max after centering (second pass)
    float2 minmax = blockwise_quantization::compute_centered_minmax_simdgroup(
        input, block_start, block_elements, K, block_mean, simd_lane_id
    );
    float centered_min = minmax.x;
    float centered_max = minmax.y;

    // Step 3: Compute scale and zero point
    float range = max(abs(centered_min), abs(centered_max));
    float scale = (range > 0.0f) ? (range / 127.0f) : 1.0f;  // Symmetric range
    int8_t zero_point = int8_t(round(-block_mean / scale));

    // Store scale and zero point (first thread only)
    if (simd_lane_id == 0) {
        blockScales[block_idx] = scale;
        blockZeroPoints[block_idx] = zero_point;
    }

    // Step 4: Quantize using asymmetric form to save subtract
    // q = clamp(round(x/scale) + z, -128, 127)
    int32_t block_sum = 0;

    // Process in vectorized chunks where possible
    for (uint i = tid * 4; i < block_elements; i += 256 * 4) {  // 256 threads per block * 4 elements
        uint base_idx = block_start + i;

        if (i + 3 < block_elements) {
            // Vectorized path
            float4 input_vec = float4(input[base_idx], input[base_idx + 1],
                                     input[base_idx + 2], input[base_idx + 3]);
            char4 quantized = blockwise_quantization::quantize_fp32_vec4_asymmetric(input_vec, scale, zero_point);

            output[base_idx] = quantized.x;
            output[base_idx + 1] = quantized.y;
            output[base_idx + 2] = quantized.z;
            output[base_idx + 3] = quantized.w;

            // Accumulate for precomputed sum
            if (blockSums) {
                block_sum += int32_t(quantized.x) + int32_t(quantized.y) +
                           int32_t(quantized.z) + int32_t(quantized.w);
            }
        } else {
            // Handle remainder elements
            for (uint j = 0; j < 4 && i + j < block_elements; j++) {
                float val = input[base_idx + j];
                int32_t quantized = int32_t(round(val / scale)) + int32_t(zero_point);
                quantized = clamp(quantized, -128, 127);
                output[base_idx + j] = int8_t(quantized);

                if (blockSums) {
                    block_sum += quantized;
                }
            }
        }
    }

    // Step 5: Optional - store precomputed block sum for weights
    if (blockSums) {
        block_sum = simd_sum(block_sum);
        if (simd_lane_id == 0) {
            atomic_fetch_add_explicit((device atomic_int*)&blockSums[block_idx],
                                    block_sum, memory_order_relaxed);
        }
    }
}

kernel void quantize_blockwise_centered_fp16_to_int8(
    device half* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device float* blockScales [[buffer(2)]],
    device int8_t* blockZeroPoints [[buffer(3)]],
    device int32_t* blockSums [[buffer(4)]], // Optional: precompute Σ Q
    constant uint& K [[buffer(5)]],
    constant uint& blockSizeK [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]
) {
    const uint block_idx = gid / blockSizeK;
    const uint num_blocks = (K + blockSizeK - 1) / blockSizeK;

    if (block_idx >= num_blocks) return;

    const uint block_start = block_idx * blockSizeK;
    const uint block_end = min(block_start + blockSizeK, K);
    const uint block_elements = block_end - block_start;

    // Step 1: Compute block mean using simdgroup operations
    // Convert FP16 to FP32 for mean computation
    float sum = 0.0f;
    uint count = 0;

    for (uint i = simd_lane_id; i < block_elements && (block_start + i) < K; i += 32) {
        sum += float(input[block_start + i]);
        count++;
    }

    sum = simd_sum(sum);
    count = simd_sum(count);
    float block_mean = (count > 0) ? (sum / float(count)) : 0.0f;

    // Step 2: Find min/max after centering
    float local_min = INFINITY;
    float local_max = -INFINITY;

    for (uint i = simd_lane_id; i < block_elements; i += 32) {
        float centered_val = float(input[block_start + i]) - block_mean;
        local_min = min(local_min, centered_val);
        local_max = max(local_max, centered_val);
    }

    float centered_min = simd_min(local_min);
    float centered_max = simd_max(local_max);

    // Step 3: Compute scale and zero point
    float range = max(abs(centered_min), abs(centered_max));
    float scale = (range > 0.0f) ? (range / 127.0f) : 1.0f;
    int8_t zero_point = int8_t(round(-block_mean / scale));

    // Store scale and zero point (first thread only)
    if (simd_lane_id == 0) {
        blockScales[block_idx] = scale;
        blockZeroPoints[block_idx] = zero_point;
    }

    // Step 4: Quantize
    int32_t block_sum = 0;

    for (uint i = tid * 4; i < block_elements; i += 256 * 4) {
        uint base_idx = block_start + i;

        if (i + 3 < block_elements) {
            half4 input_vec = half4(input[base_idx], input[base_idx + 1],
                                   input[base_idx + 2], input[base_idx + 3]);
            char4 quantized = blockwise_quantization::quantize_fp16_vec4_asymmetric(input_vec, scale, zero_point);

            output[base_idx] = quantized.x;
            output[base_idx + 1] = quantized.y;
            output[base_idx + 2] = quantized.z;
            output[base_idx + 3] = quantized.w;

            if (blockSums) {
                block_sum += int32_t(quantized.x) + int32_t(quantized.y) +
                           int32_t(quantized.z) + int32_t(quantized.w);
            }
        } else {
            for (uint j = 0; j < 4 && i + j < block_elements; j++) {
                float val = float(input[base_idx + j]);
                int32_t quantized = int32_t(round(val / scale)) + int32_t(zero_point);
                quantized = clamp(quantized, -128, 127);
                output[base_idx + j] = int8_t(quantized);

                if (blockSums) {
                    block_sum += quantized;
                }
            }
        }
    }

    // Step 5: Optional - store precomputed block sum
    if (blockSums) {
        block_sum = simd_sum(block_sum);
        if (simd_lane_id == 0) {
            atomic_fetch_add_explicit((device atomic_int*)&blockSums[block_idx],
                                    block_sum, memory_order_relaxed);
        }
    }
}

kernel void quantize_blockwise_centered_bf16_to_int8(
    device bfloat* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device float* blockScales [[buffer(2)]],
    device int8_t* blockZeroPoints [[buffer(3)]],
    device int32_t* blockSums [[buffer(4)]], // Optional: precompute Σ Q
    constant uint& K [[buffer(5)]],
    constant uint& blockSizeK [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    ushort simd_lane_id [[thread_index_in_simdgroup]]
) {
    const uint block_idx = gid / blockSizeK;
    const uint num_blocks = (K + blockSizeK - 1) / blockSizeK;

    if (block_idx >= num_blocks) return;

    const uint block_start = block_idx * blockSizeK;
    const uint block_end = min(block_start + blockSizeK, K);
    const uint block_elements = block_end - block_start;

    // Step 1: Compute block mean using simdgroup operations
    // Convert BF16 to FP32 for mean computation
    float sum = 0.0f;
    uint count = 0;

    for (uint i = simd_lane_id; i < block_elements && (block_start + i) < K; i += 32) {
        // Convert BF16 to FP32
        ushort bf16_bits = as_type<ushort>(input[block_start + i]);
        uint fp32_bits = uint(bf16_bits) << 16;
        float fp32_val = as_type<float>(fp32_bits);
        sum += fp32_val;
        count++;
    }

    sum = simd_sum(sum);
    count = simd_sum(count);
    float block_mean = (count > 0) ? (sum / float(count)) : 0.0f;

    // Step 2: Find min/max after centering
    float local_min = INFINITY;
    float local_max = -INFINITY;

    for (uint i = simd_lane_id; i < block_elements; i += 32) {
        ushort bf16_bits = as_type<ushort>(input[block_start + i]);
        uint fp32_bits = uint(bf16_bits) << 16;
        float fp32_val = as_type<float>(fp32_bits);
        float centered_val = fp32_val - block_mean;
        local_min = min(local_min, centered_val);
        local_max = max(local_max, centered_val);
    }

    float centered_min = simd_min(local_min);
    float centered_max = simd_max(local_max);

    // Step 3: Compute scale and zero point
    float range = max(abs(centered_min), abs(centered_max));
    float scale = (range > 0.0f) ? (range / 127.0f) : 1.0f;
    int8_t zero_point = int8_t(round(-block_mean / scale));

    // Store scale and zero point (first thread only)
    if (simd_lane_id == 0) {
        blockScales[block_idx] = scale;
        blockZeroPoints[block_idx] = zero_point;
    }

    // Step 4: Quantize
    int32_t block_sum = 0;

    for (uint i = tid * 4; i < block_elements; i += 256 * 4) {
        uint base_idx = block_start + i;

        if (i + 3 < block_elements) {
            ushort4 bf16_vec = ushort4(
                as_type<ushort>(input[base_idx]),
                as_type<ushort>(input[base_idx + 1]),
                as_type<ushort>(input[base_idx + 2]),
                as_type<ushort>(input[base_idx + 3])
            );

            char4 quantized = blockwise_quantization::quantize_bf16_vec4_asymmetric(bf16_vec, scale, zero_point);

            output[base_idx] = quantized.x;
            output[base_idx + 1] = quantized.y;
            output[base_idx + 2] = quantized.z;
            output[base_idx + 3] = quantized.w;

            if (blockSums) {
                block_sum += int32_t(quantized.x) + int32_t(quantized.y) +
                           int32_t(quantized.z) + int32_t(quantized.w);
            }
        } else {
            for (uint j = 0; j < 4 && i + j < block_elements; j++) {
                ushort bf16_bits = as_type<ushort>(input[base_idx + j]);
                uint fp32_bits = uint(bf16_bits) << 16;
                float val = as_type<float>(fp32_bits);
                int32_t quantized = int32_t(round(val / scale)) + int32_t(zero_point);
                quantized = clamp(quantized, -128, 127);
                output[base_idx + j] = int8_t(quantized);

                if (blockSums) {
                    block_sum += quantized;
                }
            }
        }
    }

    // Step 5: Optional - store precomputed block sum
    if (blockSums) {
        block_sum = simd_sum(block_sum);
        if (simd_lane_id == 0) {
            atomic_fetch_add_explicit((device atomic_int*)&blockSums[block_idx],
                                    block_sum, memory_order_relaxed);
        }
    }
}
//
//  GEMMBFloatTypes.h
//  FlashAttention
//
//  Defines bfloat vector types for Metal (which only has bfloat scalar type)
//

#ifndef __GEMM_BFLOAT_TYPES_H
#define __GEMM_BFLOAT_TYPES_H

#include <metal_stdlib>
using namespace metal;

#if !defined(__HAVE_BFLOAT__)
// Provide a software fallback for bfloat when the toolchain lacks native support.
// The implementation keeps IEEE semantics by round-tripping through float.
namespace mfa_detail {
inline ushort float_to_bfloat_bits(float value) {
    uint bits = as_type<uint>(value);
    // Round to nearest even by adding 0x7FFF plus the LSB of the target mantissa.
    uint rounding_bias = 0x7FFF + ((bits >> 16) & 1u);
    return ushort((bits + rounding_bias) >> 16);
}

inline float bfloat_bits_to_float(ushort bits) {
    uint widened = uint(bits) << 16;
    return as_type<float>(widened);
}
}  // namespace mfa_detail

struct alignas(2) bfloat {
    ushort storage;

    bfloat() = default;
    bfloat(const bfloat&) = default;
    bfloat& operator=(const bfloat&) = default;

    // Allow construction from common scalar types used in kernels.
    bfloat(float value) : storage(mfa_detail::float_to_bfloat_bits(value)) {}
    bfloat(half value) : storage(mfa_detail::float_to_bfloat_bits(float(value))) {}
    bfloat(int value) : storage(mfa_detail::float_to_bfloat_bits(float(value))) {}
    bfloat(uint value) : storage(mfa_detail::float_to_bfloat_bits(float(value))) {}

    bfloat& operator=(float value) {
        storage = mfa_detail::float_to_bfloat_bits(value);
        return *this;
    }

    bfloat& operator=(half value) {
        storage = mfa_detail::float_to_bfloat_bits(float(value));
        return *this;
    }

    bfloat& operator=(int value) {
        storage = mfa_detail::float_to_bfloat_bits(float(value));
        return *this;
    }

    operator float() const {
        return mfa_detail::bfloat_bits_to_float(storage);
    }

    operator half() const {
        return half(mfa_detail::bfloat_bits_to_float(storage));
    }
};

// Define bfloat vector types (Metal only has bfloat scalar type on older toolchains)
struct packed_bfloat2 {
    bfloat x, y;

    packed_bfloat2() = default;
    packed_bfloat2(bfloat x_, bfloat y_) : x(x_), y(y_) {}

    // Subscript operator for element access
    thread bfloat& operator[](int i) thread {
        return (i == 0) ? x : y;
    }

    thread const bfloat& operator[](int i) const thread {
        return (i == 0) ? x : y;
    }
};

struct packed_bfloat4 {
    bfloat x, y, z, w;

    packed_bfloat4() = default;
    packed_bfloat4(bfloat x_, bfloat y_, bfloat z_, bfloat w_) : x(x_), y(y_), z(z_), w(w_) {}

    // Allow conversion to float4 for quantization
    operator float4() const {
        return float4(float(x), float(y), float(z), float(w));
    }

    // Subscript operator for element access
    thread bfloat& operator[](int i) thread {
        switch(i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: return w;
        }
    }

    thread const bfloat& operator[](int i) const thread {
        switch(i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: return w;
        }
    }
};

// Alias for consistency with other vector types
typedef packed_bfloat2 bfloat2;
typedef packed_bfloat4 bfloat4;
#endif // !defined(__HAVE_BFLOAT__)

#endif // __GEMM_BFLOAT_TYPES_H

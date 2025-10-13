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

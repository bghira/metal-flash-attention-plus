//
//  GEMMBFloatTypes.h
//  FlashAttention
//
//  Ensures the Metal toolchain provides native bfloat support.
//

#ifndef __GEMM_BFLOAT_TYPES_H
#define __GEMM_BFLOAT_TYPES_H

#include <metal_stdlib>
using namespace metal;

#if !defined(__HAVE_BFLOAT__)
#error "Metal compiler must provide native bfloat support (macOS 15 or newer)."
#endif

#endif // __GEMM_BFLOAT_TYPES_H

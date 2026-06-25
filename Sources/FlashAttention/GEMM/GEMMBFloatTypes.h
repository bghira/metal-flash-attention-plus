//
//  GEMMBFloatTypes.h
//  FlashAttention
//
//  Ensures the Metal toolchain provides native bfloat support, or falls back
//  to half-precision aliases so generated kernels still compile/run on
//  toolchains whose runtime compiler (device.makeLibrary) does not expose
//  native bfloat (e.g. some macOS 15 CI runner images). Native bfloat is
//  used wherever __HAVE_BFLOAT__ is defined; the fallback is numerically
//  approximate (half instead of bfloat) but keeps the kernels buildable.
//

#ifndef __GEMM_BFLOAT_TYPES_H
#define __GEMM_BFLOAT_TYPES_H

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)
  // Native bfloat is available — nothing to do; the built-in types are used.
#elif defined(__METAL_VERSION__)
  // No native bfloat: alias to half so the generated bfloat load/store paths
  // compile. Results differ from true bfloat rounding, which is acceptable for
  // environments that cannot compile native bfloat at all.
  using bfloat = half;
  using bfloat2 = half2;
  using bfloat3 = half3;
  using bfloat4 = half4;
  using packed_bfloat2 = packed_half2;
  using packed_bfloat4 = packed_half4;
#else
  #error "GEMMBFloatTypes.h must be included in a Metal translation unit."
#endif

#endif // __GEMM_BFLOAT_TYPES_H

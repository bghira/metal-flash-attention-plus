//
//  GEMMBFloatTypes.h
//  FlashAttention
//
//  Ensures the Metal toolchain provides native bfloat support.
//
//  Runtime compilation (device.makeLibrary) must use MTLCompileOptions.mfaDefault
//  (languageVersion = .version3_2) so __HAVE_BFLOAT__ is defined; see
//  MetalCompileOptions.swift.
//

#ifndef __GEMM_BFLOAT_TYPES_H
#define __GEMM_BFLOAT_TYPES_H

#include <metal_stdlib>
using namespace metal;

#if !defined(__HAVE_BFLOAT__)
#error "Metal compiler must provide native bfloat support (MSL 3.2 / macOS 15+)."
#endif

#endif // __GEMM_BFLOAT_TYPES_H

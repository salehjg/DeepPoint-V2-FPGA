/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch) with small changes to the original design.
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <type_traits>
#include "AxiHelper.h"
#include "xilinx/config.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Resource.h"
#include "hlslib/xilinx/Stream.h"

using hlslib::Stream;
using namespace ConfigTaskConv2;

constexpr int kSeed = 5; // For initializing matrices for testing
constexpr unsigned kPipeDepth = 4;

// Memory bus in K-dimension
constexpr int kMemoryWidthK = kMemoryWidthBytesK / sizeof(CONFIG_DTYPE);
static_assert(kMemoryWidthBytesK % sizeof(CONFIG_DTYPE) == 0,
                            "Memory width in K not divisible by size of data type.");
using MemoryPackK_t = hlslib::DataPack<CONFIG_DTYPE, kMemoryWidthK>;

// Memory bus in M-dimension
constexpr int kMemoryWidthM = kMemoryWidthBytesM / sizeof(CONFIG_DTYPE);
static_assert(kMemoryWidthBytesM % sizeof(CONFIG_DTYPE) == 0,
                            "Memory width in M not divisible by size of data type.");
using MemoryPackM_t = hlslib::DataPack<CONFIG_DTYPE, kMemoryWidthM>;

// Internal compute buses
using ComputePackN_t = hlslib::DataPack<CONFIG_DTYPE, kComputeTileSizeN>;
using ComputePackM_t = hlslib::DataPack<CONFIG_DTYPE, kComputeTileSizeM>;
using OutputPack_t = hlslib::DataPack<CONFIG_DTYPE, kComputeTileSizeM>;

// On-chip transpose of A
constexpr int kTransposeWidth = kTransposeWidthBytes / sizeof(CONFIG_DTYPE);
static_assert(kTransposeWidthBytes % sizeof(CONFIG_DTYPE) == 0,
                            "Transpose width must be divisible by data size.");
static_assert(kTransposeWidthBytes % kMemoryWidthBytesK == 0,
                            "Transpose width must be divisible by memory port width.");

using MemoryPackA_t = MemoryPackK_t;
constexpr decltype(kMemoryWidthK) kMemoryWidthA = kMemoryWidthK;

constexpr unsigned long kOuterTileSizeMMemory = kOuterTileSizeM / kMemoryWidthM;
static_assert(
        kOuterTileSizeM % kMemoryWidthM == 0,
        "Outer memory tile size in M must be divisible by memory port width.");

constexpr unsigned long kInnerTilesN = kOuterTileSizeN / kInnerTileSizeN;
static_assert(kOuterTileSizeN % kInnerTileSizeN == 0,
                            "Outer tile size must be divisible by the inner tile size.");

constexpr unsigned long kInnerTilesM = kOuterTileSizeM / kComputeTileSizeM;
static_assert(kOuterTileSizeM % kComputeTileSizeM == 0,
                            "Outer tile size must be divisible by compute tile size in M.");

constexpr unsigned long kComputeTilesN = kInnerTileSizeN / kComputeTileSizeN;
static_assert(kInnerTileSizeN % kComputeTileSizeN == 0,
                            "Inner tile size must be divisible by compute tile size.");

inline unsigned SizeKMemory(unsigned k) {
    #pragma HLS INLINE
    return DivCeil<unsigned>(k, kMemoryWidthK);
}

inline unsigned SizeMMemory(unsigned m) {
    #pragma HLS INLINE
    return m / kMemoryWidthM;
}

inline unsigned OuterTilesN(unsigned n) {
    #pragma HLS INLINE
    return n / kOuterTileSizeN;
}

inline unsigned OuterTilesM(unsigned m) {
    #pragma HLS INLINE
    return m / kOuterTileSizeM;
}

inline unsigned long TotalReadsFromA(const unsigned size_n,
                                     const unsigned size_k,
                                     const unsigned size_m) {
    #pragma HLS INLINE
    return static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
                 kOuterTileSizeN * ( DivCeil<unsigned>(size_k, kTransposeWidth)*kTransposeWidth );
}

inline unsigned long TotalReadsFromB(const unsigned size_n,
                                     const unsigned size_k,
                                     const unsigned size_m) {
    #pragma HLS INLINE
    return static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
                 kOuterTileSizeM * size_k;
}

template <typename T,
        class = typename std::enable_if<std::is_integral<T>::value, T>::type>
    constexpr T PowerOfTwo(T number, unsigned char power) {
    return (number > 0) ? PowerOfTwo(number >> 1, power + 1) : (1 << (power - 1));
}

#ifdef MM_ADD_RESOURCE
#define MM_ADD_RESOURCE_PRAGMA(var) \
    HLSLIB_RESOURCE_PRAGMA(var, MM_ADD_RESOURCE)
#else
#define MM_ADD_RESOURCE_PRAGMA(var)
#endif

#ifdef MM_MULT_RESOURCE
#define MM_MULT_RESOURCE_PRAGMA(var) \
    HLSLIB_RESOURCE_PRAGMA(var, MM_MULT_RESOURCE)
#else
#define MM_MULT_RESOURCE_PRAGMA(var)
#endif
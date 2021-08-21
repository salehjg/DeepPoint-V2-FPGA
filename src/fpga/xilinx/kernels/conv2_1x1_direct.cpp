/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch) 
///            with small changes to the original design by Saleh Jamali Golzar.
/// @date      June 2018 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include <cassert>
#include <iostream>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "AxiHelper.h"
#include "Conv2D.h"
#include "xilinx/config.h"

using namespace ConfigTaskConv2;

void ProcessingElement( Stream<ComputePackN_t, kPipeDepth> &aIn,
                        Stream<ComputePackN_t, kPipeDepth> &aOut,
                        Stream<ComputePackM_t, kPipeDepth> &bIn,
                        Stream<ComputePackM_t, kPipeDepth> &bOut,
                        Stream<ComputePackM_t> &cOut,
                        Stream<ComputePackM_t> &cIn, const unsigned locationN,
                        const unsigned size_n, const unsigned size_k,
                        const unsigned size_m) {

    assert((static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) * size_k *
            kInnerTilesN * kInnerTilesM * kComputeTileSizeN *
            kComputeTileSizeM) ==
            ((static_cast<unsigned long>(size_n) * size_k * size_m) / kComputeTilesN));

    // A is double-buffered, such that new values can be read while the 
    // previous outer product is being computed. This is required to achieve
    // a perfect pipeline across the K-dimension, which is necessary for
    // many processing elements (kInnerTileSizeN).
    ComputePackN_t aBuffer[2 * kInnerTilesN];

    // This is where we spend all our T^2 fast memory
    ComputePackM_t cBuffer[kInnerTilesN * kInnerTilesM][kComputeTileSizeN];
#pragma HLS ARRAY_PARTITION variable=cBuffer complete dim=2

    // Populate the buffer for the first outer product 
    InitializeABuffer_Inner:
    for (unsigned n2 = 0; n2 < kInnerTilesN; ++n2) {
        if (locationN < kComputeTilesN - 1) {
            // All but the last processing element 
            InitializeABuffer_Outer:
            for (unsigned n1 = 0; n1 < kComputeTilesN - locationN; ++n1) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
                const auto read = aIn.Pop();
                if (n1 == 0) {
                    aBuffer[n2] = read;
                } else {
                    aOut.Push(read);
                }
            }
        } else {
            // Last processing element gets a special case, because Vivado HLS
            // refuses to flatten and pipeline loops with trip count 1
#pragma HLS PIPELINE II=1
            aBuffer[n2] = aIn.Pop();
        }
    }

    OuterTile_N:
    for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
    OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {

    // We do not tile K further, but loop over the entire outer tile here
    Collapse_K:
    for (unsigned k = 0; k < size_k; ++k) {
        // Begin outer tile ---------------------------------------------------

        Pipeline_N:
        for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {

            Pipeline_M:
            for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {

                // Begin compute tile ---------------------------------------------
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN

                static_assert(kInnerTilesM >= kInnerTilesN,
                "Double buffering does not work if there are more "
                "N-tiles than M-tiles");

                // Double-buffering scheme. This hijacks the m1-index to perform
                // the buffering and forwarding of values for the following outer
                // product, required to flatten the K-loop.
                if ((n0 < OuterTilesN(size_n) - 1 || m0 < OuterTilesM(size_m) - 1 ||
                    k < size_k - 1) &&
                    m1 >= locationN            // Start at own index.
                    && m1 < kComputeTilesN) {  // Number of PEs in front.

                    const auto read = aIn.Pop();
                    if (m1 == locationN) {

                        // Double buffering
                        aBuffer[n1 + (k % 2 == 0 ? kInnerTilesN : 0)] = read;
#pragma HLS DEPENDENCE variable=aBuffer false
                    } else {
                        // Without this check, Vivado HLS thinks aOut can be written
                        // from the last processing element and fails dataflow
                        // checking.
                        if (locationN < kComputeTilesN - 1) {
                            aOut.Push(read);
                        }
                    }

                }

                // Double buffering, read from the opposite end of where the buffer
                // is being written
                const auto aVal = aBuffer[n1 + (k % 2 == 0 ? 0 : kInnerTilesN)];
#pragma HLS DEPENDENCE variable=aBuffer false
                const auto bVal = bIn.Pop();
                if (locationN < kComputeTilesN - 1) {
                    bOut.Push(bVal);
                }

                Unroll_N:
                for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
                    #pragma HLS UNROLL

                    ComputePackM_t cStore;
                    const auto cPrev = (k > 0)? cBuffer[n1 * kInnerTilesM + m1][n2]
                                            : ComputePackM_t(static_cast<CONFIG_DTYPE>(0));

                    Unroll_M:
                    for (unsigned m2 = 0; m2 < kComputeTileSizeM; ++m2) {
#pragma HLS UNROLL
                        const auto mapped = OperatorMap::Apply(aVal[n2], bVal[m2]);
                        MM_MULT_RESOURCE_PRAGMA(mapped);
                        const auto prev = cPrev[m2];

                        const auto reduced = OperatorReduce::Apply(prev, mapped);
                        MM_ADD_RESOURCE_PRAGMA(reduced);
                        cStore[m2] = reduced; 
#pragma HLS DEPENDENCE variable=cBuffer false
                    }

                    cBuffer[n1 * kInnerTilesM + m1][n2] = cStore;
                }

            // End compute tile -----------------------------------------------
            }
        }

        // End outer tile -----------------------------------------------------
    }

    // Write back tile of C -------------------------------------------------
    // 
    // This uses a flattened implementation of the loops, as we otherwise
    // introduce a lot of pipeline drains, which can have a small performance
    // impact for large designs.
    //
    const unsigned writeFlattenedInner =
                (kComputeTileSizeN * kInnerTilesM +
                (kComputeTilesN - locationN - 1) * kComputeTileSizeN * kInnerTilesM);
    const unsigned writeFlattened = kInnerTilesN * writeFlattenedInner;
    ap_uint<hlslib::ConstLog2(kInnerTilesN)> n1 = 0;
    ap_uint<hlslib::ConstLog2(kComputeTileSizeN)> n2 = 0;
    ap_uint<hlslib::ConstLog2(kInnerTilesM)> m1 = 0;
    unsigned inner = 0;

    WriteC_Flattened:
    for (unsigned i = 0; i < writeFlattened; ++i) {
#pragma HLS PIPELINE II=1
        if (inner < kComputeTileSizeN * kInnerTilesM) {
            cOut.Push(cBuffer[n1 * kInnerTilesM + m1][n2]);
            if (m1 == kInnerTilesM - 1) {
                m1 = 0;
                if (n2 == kComputeTileSizeN - 1) {
                    n2 = 0;
                } else {
                    ++n2;
                }
            } else {
                ++m1;
            }
        } else {
            if (locationN < kComputeTilesN - 1) {
                cOut.Push(cIn.Pop());
            }
        }
        if (inner == writeFlattenedInner - 1) {
            inner = 0;
            ++n1;
        } else {
            ++inner;
        }
    }

    }
    }

}

unsigned IndexA(const unsigned n0, const unsigned n1, const unsigned n2,
                const unsigned k0, const unsigned k1, const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {
#pragma HLS INLINE
    const auto index =
            (n0 * kOuterTileSizeN + n1 * kInnerTileSizeN + n2) * SizeKMemory(size_k) +
            (k0 * (kTransposeWidth / kMemoryWidthK) + k1);
    assert(index < size_n * SizeKMemory(size_k));
    return index;
}

unsigned IndexB(const unsigned k, const unsigned m0, const unsigned m1m,
                const unsigned size_n, const unsigned size_k,
                const unsigned size_m) {
#pragma HLS INLINE
    const auto index =
            k * SizeMMemory(size_m) + (m0 * kOuterTileSizeMMemory + m1m);
    assert(index < size_k * SizeMMemory(size_m));
    return index;
}

unsigned IndexC(const unsigned n0, const unsigned n1, const unsigned m0,
                const unsigned m1m, const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {
#pragma HLS INLINE
    const auto index = (n0 * kOuterTileSizeN + n1) * SizeMMemory(size_m) +
            (m0 * kOuterTileSizeMMemory + m1m);
    assert(index < size_n * SizeMMemory(size_m));
    return index;
}

void _ReadAInner(MemoryPackK_t const a[],
                Stream<CONFIG_DTYPE, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                const unsigned n0, const unsigned n1, const unsigned n2,
                const unsigned k0, const unsigned k1, const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {
#pragma HLS INLINE
    auto pack = a[IndexA(n0, n1, n2, k0, k1, size_n, size_k, size_m)];
#ifdef KERNEL_LOGS
    //std::cout << "ReadA: index=" << IndexA(n0, n1, n2, k0, k1, size_n, size_k, size_m) << "\n";
#endif
    ReadA_Unroll:
    for (unsigned w = 0; w < kMemoryWidthK; ++w) {
#pragma HLS UNROLL
        aSplit[k1 * kMemoryWidthK + w].Push(pack[w]); 
    }
}

template <unsigned innerReads>
void _ReadAInnerLoop(
                MemoryPackK_t const a[],
                Stream<CONFIG_DTYPE, 2 * kOuterTileSizeN> aSplit[kTransposeWidth], 
                unsigned n0, unsigned n1, unsigned n2, unsigned k0,
                const unsigned size_n, const unsigned size_k, const unsigned size_m) {
#pragma HLS INLINE
    ReadA_TransposeWidth:
    for (unsigned k1 = 0; k1 < (kTransposeWidth / kMemoryWidthK); ++k1) { 
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        _ReadAInner(a, aSplit, n0, n1, n2, k0, k1, size_n, size_k, size_m);
    }
}

// Need a special case for kMemoryWidthK == kTransposeWidth, as Vivado HLS
// otherwise doesn't pipeline the loops (because the inner trip count is 1).
template <>
void _ReadAInnerLoop<1>(
                MemoryPackK_t const a[],
                Stream<CONFIG_DTYPE, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                const unsigned n0, const unsigned n1, const unsigned n2, 
                const unsigned k0, const unsigned size_n, const unsigned size_k,
                const unsigned size_m) {
#pragma HLS INLINE
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
    _ReadAInner(a, aSplit, n0, n1, n2, k0, 0, size_n, size_k, size_m);
}

void ReadA(MemoryPackK_t const a[],
                Stream<CONFIG_DTYPE, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                const unsigned size_n, const unsigned size_k,
                const unsigned size_m) {

    const auto size_k_burst_count = DivCeil<unsigned>(size_k, kTransposeWidth);

    assert((static_cast<unsigned long>(OuterTilesN(size_n)) *
            OuterTilesM(size_m) * DivCeil<unsigned>(size_k, kTransposeWidth) * kInnerTilesN *
            kInnerTileSizeN * (kTransposeWidth / kMemoryWidthK) *
            MemoryPackK_t::kWidth) == TotalReadsFromA(size_n, size_k, size_m));

    ReadA_N0:
    for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
        ReadA_M0:
        for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
            ReadA_K0:
            for (unsigned k0 = 0; k0 < size_k_burst_count; ++k0) { // was size_k / kTransposeWidth
                ReadA_N1:
                for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {
                    ReadA_N2:
                    for (unsigned n2 = 0; n2 < kInnerTileSizeN; ++n2) {
                        _ReadAInnerLoop<kTransposeWidth / kMemoryWidthK>(
                            a, aSplit, n0, n1, n2, k0, size_n, size_k, size_m);
                    }
                }
            }
        }
    }

}

template <unsigned inner_tiles>
void _TransposeAInner(
                Stream<CONFIG_DTYPE, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                Stream<ComputePackN_t, kPipeDepth> &toKernel, const unsigned k,
                const unsigned size_k) {
#pragma HLS INLINE
    for (unsigned n1 = 0; n1 < kOuterTileSizeN / kComputeTileSizeN; ++n1) {
        ComputePackN_t pack;
        TransposeA_N2:
        for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
            pack[n2] = aSplit[k % kTransposeWidth].Pop();
            // Pop from each stream kOuterTileSizeN times in a row
            if (n2 == kComputeTileSizeN - 1) {
                toKernel.Push(pack);
            }
        }
    }
}

template <>
void _TransposeAInner<1>(
                Stream<CONFIG_DTYPE, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                Stream<ComputePackN_t, kPipeDepth> &toKernel, 
                const unsigned k, const unsigned size_k) {
#pragma HLS INLINE
    for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        ComputePackN_t pack;
        pack[0] = aSplit[k % kTransposeWidth].Pop();
        if(k<size_k){
            toKernel.Push(pack);
        }
    }
}

// We pop from the column buffers in column-major order, funneling the
// transposed data to the kernel
void TransposeA(Stream<CONFIG_DTYPE, 2 * kOuterTileSizeN> aSplit[kTransposeWidth],
                Stream<ComputePackN_t, kPipeDepth> &toKernel,
                const unsigned size_n, const unsigned size_k,
                const unsigned size_m) {

    assert((static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) * 
            DivCeil<unsigned>(size_k, kTransposeWidth) * kTransposeWidth * 
            kOuterTileSizeN) == TotalReadsFromA(size_n, size_k, size_m));

    const auto size_k_words = DivCeil<unsigned>(size_k, kTransposeWidth);
    const auto size_k_padded = size_k_words * kTransposeWidth;

    TransposeA_N0:
    for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
        TransposeA_M0:
        for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
            TransposeA_K:
            for (unsigned k = 0; k < size_k_padded; ++k) {
                _TransposeAInner<kComputeTileSizeN>(aSplit, toKernel, k, size_k);
            }
        }
    }
}

void ReadB(MemoryPackM_t const memory[],
                Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> &pipe,
                const unsigned size_n, const unsigned size_k,
                const unsigned size_m) {

    assert((static_cast<unsigned long>(OuterTilesN(size_n)) *
    OuterTilesM(size_m) * size_k * kOuterTileSizeMMemory *
    MemoryPackM_t::kWidth) == TotalReadsFromB(size_n, size_k, size_m));

    ReadB_OuterTile_N:
    for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
        ReadB_OuterTile_M:
        for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
            ReadB_K:
            for (unsigned k = 0; k < size_k; ++k) {

                ReadB_BufferB_M1:
                for (unsigned m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
                    pipe.Push(memory[IndexB(k, m0, m1m, size_n, size_k, size_m)]);
#ifdef KERNEL_LOGS
                    std::cout << "ReadB: index=" << IndexB(k, m0, m1m, size_n, size_k, size_m) << "\n";
#endif
                }

            }
        }
    }
}

void ConvertWidthB(Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> &wide,
                Stream<ComputePackM_t> &narrow, const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {

    assert(kMemoryWidthM % kComputeTileSizeM == 0);

    assert(((TotalReadsFromB(size_n, size_k, size_m) / kMemoryWidthM) *
            MemoryPackM_t::kWidth) == TotalReadsFromB(size_n, size_k, size_m));

    assert(((TotalReadsFromB(size_n, size_k, size_m) / kMemoryWidthM) *
            (kMemoryWidthM / kComputeTileSizeM) * ComputePackM_t::kWidth) ==
            TotalReadsFromB(size_n, size_k, size_m));

    ConvertWidthB_Outer:
    for (
        unsigned i = 0; 
        i < TotalReadsFromB(size_n, size_k, size_m) / kMemoryWidthM;
        ++i) {

        MemoryPackM_t memoryPack;
        ConvertWidthB_Memory:
        for (unsigned j = 0; j < kMemoryWidthM / kComputeTileSizeM; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
            if (j == 0) {
                memoryPack = wide.Pop();
            }
            ComputePackM_t computePack;
            ConvertWidthB_Compute:
            for (unsigned w = 0; w < kComputeTileSizeM; ++w) {
#pragma HLS UNROLL
                computePack[w] = memoryPack[j * kComputeTileSizeM + w];
            }
            narrow.Push(computePack);
        }
    }
}

void ConvertWidthC(Stream<ComputePackM_t> &narrow,
                Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> &wide,
                const unsigned size_n, const unsigned size_k,
                const unsigned size_m) {

    assert(kMemoryWidthM % ComputePackM_t::kWidth == 0);

    assert((((size_n * size_m) / MemoryPackM_t::kWidth) *
            (kMemoryWidthM / ComputePackM_t::kWidth) * ComputePackM_t::kWidth) ==
            size_n * size_m);

    ConvertWidthC_Outer:
    for (unsigned i = 0; i < (size_n * size_m) / MemoryPackM_t::kWidth; ++i) {
#ifdef MM_CONVERT_B
        ConvertWidthB_Memory:
        MemoryPackM_t memoryPack;
        for (unsigned j = 0; j < kMemoryWidthM / ComputePackM_t::kWidth; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
            const auto computePack = narrow.Pop();
            ConvertWidthB_Compute:
            for (unsigned w = 0; w < ComputePackM_t::kWidth; ++w) {
#pragma HLS UNROLL
                memoryPack[j * ComputePackM_t::kWidth + w] = computePack[w];
            }
            if (j == kMemoryWidthM / ComputePackM_t::kWidth - 1) {
                wide.Push(memoryPack);
            }
        }
#else
        wide.Push(narrow.Pop());
#endif
    }
}

void WriteC(Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> &pipe,
                MemoryPackM_t const bias[],
                MemoryPackM_t memory[], const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {



    assert((OuterTilesN(size_n) * OuterTilesM(size_m) * kOuterTileSizeN *
    kOuterTileSizeMMemory * MemoryPackM_t::kWidth) == size_n * size_m);

    //------------Additional Code for BiasTn Addition-------------------
    constexpr unsigned MaxSizeM = 1024;
    assert(size_m % kMemoryWidthM==0);
    assert(MaxSizeM>=size_m);
    MemoryPackM_t biasBuff[MaxSizeM/kMemoryWidthM];
    const auto vecCount = size_m/kMemoryWidthM;
    WriteC_LoadBiasTn_Iter:
    for(unsigned iter=0; iter<vecCount; iter++){
#pragma HLS PIPELINE II=1
        biasBuff[iter] = bias[iter];
#ifdef KERNEL_LOGS
                    //std::cout << "WriteC: Index Init Bias = " << iter << "\n";
#endif
    }
    WriteC_LoadBiasTn_Init:
    for(unsigned iter=vecCount; iter<MaxSizeM/kMemoryWidthM; iter++){
#pragma HLS PIPELINE II=1
        biasBuff[iter] = MemoryPackM_t(0.);
    }
    unsigned indxB;

    //------------------------------------------------------------------
    unsigned indxC;

    WriteC_OuterTile_N:
    for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
        WriteC_OuterTile_M:
        for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
            WriteC_N1:
            for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1) {
                WriteC_M1:
                for (unsigned m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
                    indxB = m0*kOuterTileSizeMMemory+m1m; //biasBuff's index
                    indxC = IndexC(n0, n1, m0, m1m, size_n, size_k, size_m);
                    memory[indxC] = pipe.Pop() + biasBuff[indxB];
#ifdef KERNEL_LOGS
                    //std::cout << "WriteC: IndexC = " << indxC << ", indxB:" << indxB << "\n";
#endif
                }
            }
#ifdef KERNEL_LOGS
            std::cout << "Finished tile (" << n0 << ", " << m0 << ") of ("
                    << OuterTilesN(size_n) - 1 << ", " << OuterTilesM(size_m) - 1 << ")\n"
                    << std::flush;
#endif
        }
    }
}

void FeedB(Stream<ComputePackM_t> &fromMemory,
                Stream<ComputePackM_t, kPipeDepth> &toKernel, const unsigned size_n,
                const unsigned size_k, const unsigned size_m) {
    assert(static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
            size_k * kInnerTilesM * ComputePackM_t::kWidth ==
    TotalReadsFromB(size_n, size_k, size_m));

    const unsigned bound_n = OuterTilesN(size_n);
    const unsigned bound_m = OuterTilesM(size_m);

    FeedB_OuterTile_N:
    for (unsigned n0 = 0; n0 < bound_n; ++n0) {
        FeedB_OuterTile_M:
        for (unsigned m0 = 0; m0 < bound_m; ++m0) {
            FeedB_K:
            for (unsigned k = 0; k < size_k; ++k) {

                ComputePackM_t buffer[kInnerTilesM];

                FeedB_Pipeline_N:
                for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {
                    FeedB_Pipeline_M:
                    for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
                        ComputePackM_t val;
                        if (n1 == 0) {
                            val = fromMemory.Pop();
                            buffer[m1] = val;
                        } else {
                            val = buffer[m1];
                        }
                        toKernel.Push(val);
                    }
                }

            }
        }
    }
}

extern "C" {
void task_conv2_1x1_direct(
                MemoryPackK_t const a[], //inputTn
                MemoryPackM_t const b[], //weightTn
                MemoryPackM_t const e[], //biasTn
                MemoryPackM_t c[], //outputTn
                const unsigned size_n, 
                const unsigned size_k,
                const unsigned size_m) {

#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi port=e offset=slave bundle=gmem2 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem3 max_read_burst_length=2 max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=e bundle=control
#pragma HLS INTERFACE s_axilite port=c bundle=control
#pragma HLS INTERFACE s_axilite port=size_n bundle=control
#pragma HLS INTERFACE s_axilite port=size_k bundle=control
#pragma HLS INTERFACE s_axilite port=size_m bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    Stream<CONFIG_DTYPE, 2 * kOuterTileSizeN> aSplit[kTransposeWidth];
#pragma HLS STREAM variable=aSplit depth=2*kOuterTileSizeN
    Stream<CONFIG_DTYPE> aConvert("aConvert");
    Stream<ComputePackN_t, kPipeDepth> aPipes[kComputeTilesN + 1];

    // Memory accesses and pipes for B 
    Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> bMemory("bMemory");
    Stream<ComputePackM_t, kPipeDepth> bPipes[kComputeTilesN + 1];

    // Pipes for C
    Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];

#ifndef HLSLIB_SYNTHESIS
    // Name the arrays of channels for debugging purposes
    for (unsigned i = 0; i < kTransposeWidth; ++i) {
        aSplit[i].set_name(("aSplit[" + std::to_string(i) + "]").c_str());
    }
    for (unsigned n = 0; n < kComputeTilesN; ++n) {
        aPipes[n].set_name(("aPipes[" + std::to_string(n) + "]").c_str());
    }
    for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
        bPipes[n].set_name(("bPipes[" + std::to_string(n) + "]").c_str());
    }
    for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
        cPipes[n].set_name(("cPipes[" + std::to_string(n) + "]").c_str());
    }
#endif
#ifdef KERNEL_LOGS
    std::cout<<"Simulation mode is enabled."<<std::endl;
#endif

    HLSLIB_DATAFLOW_INIT();

    // Only convert memory width if necessary
    HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aSplit, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aPipes[0], size_n, size_k, size_m);

    HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory, size_n, size_k, size_m);

    // Only convert memory width if necessary
    Stream<ComputePackM_t> bFeed("bFeed");
    HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(FeedB, bFeed, bPipes[0], size_n, size_k, size_m);

    for (unsigned pe = 0; pe < kComputeTilesN; ++pe) {
#pragma HLS UNROLL
        HLSLIB_DATAFLOW_FUNCTION(ProcessingElement,
        aPipes[pe],
        aPipes[pe + 1],
        bPipes[pe],
        bPipes[pe + 1],
        cPipes[pe],
        cPipes[pe + 1],
        pe, size_n, size_k, size_m);
    }

    Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> cMemory("cMemory");
    HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes[0], cMemory, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, e, c, size_n, size_k, size_m);

    HLSLIB_DATAFLOW_FINALIZE();
}
}


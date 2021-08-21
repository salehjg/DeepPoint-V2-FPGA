#include <cassert>
#include <iostream>
#include <limits>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Operators.h"
#include "hlslib/xilinx/TreeReduce.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskMatMul;

// Architecture adopted from https://github.com/spcl/hls_tutorial_examples

/**
 * @brief      Computes batch-matmul operation of C=AB
 *             The inputs should be of rank three and be padded in the last dimension.
 *             The latency will be reported for Shape1=5x1024x64 and Shape2=5x64x1024 and RowTileSizeD=4.
 *             This kernel complies with the padded last dim policy.
 *
 * @param[in]  A          The Input Matrix A (Rank3, RowMajor, Batch*N*K)
 * @param[in]  B          The Input Matrix B (Rank3, RowMajor, Batch*K*M)
 * @param      C          The output Matrix C(Rank3, RowMajor, Batch*N*M)
 * @param[in]  sizeBatch  Batch Size
 * @param[in]  sizeN      N 
 * @param[in]  sizeK      K
 * @param[in]  sizeM      M
 *
 */
void MatmulReorderedVectorized_V1(
    const CONFIG_DTYPE* A,
    const MemoryPackF_t* B,
    MemoryPackF_t *C,
    const unsigned sizeBatch,
    const unsigned sizeN,
    const unsigned sizeK,
    const unsigned sizeM){

    #pragma HLS INLINE

    // MatA's  shape = [dim0, dim1, dim2] = [batchSize, sizeN, sizeK] = [Batch, Height, Width]; Row-major
    // MatB's  shape = [dim0, dim1, dim2] = [batchSize, sizeK, sizeM] = [Batch, Height, Width]; Row-major
    // MatC=AB shape = [dim0, dim1, dim2] = [batchSize, sizeN, sizeM] = [Batch, Height, Width]; Row-major
 
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    const unsigned lastDimPaddedA = MakeDivisible<unsigned>(sizeK, CONFIG_M_AXI_WIDTH);
    const unsigned lastDimPaddedB = MakeDivisible<unsigned>(sizeM, CONFIG_M_AXI_WIDTH);
    const unsigned lastDimPaddedC = lastDimPaddedB;


    const unsigned vecsPerSliceA = lastDimPaddedA/CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceB = lastDimPaddedB/CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceC = lastDimPaddedC/CONFIG_M_AXI_WIDTH;

    const unsigned boundLoopN = DivCeil<unsigned>(sizeN, RowTileSizeD);

    LoopBatch:
    for(unsigned batch=0; batch<sizeBatch; batch++) {
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopN:
        for (unsigned n = 0; n < boundLoopN; n++) {
            #pragma HLS LOOP_TRIPCOUNT min=256 max=256
            MemoryPackF_t acc[RowTileSizeD][MaxM / CONFIG_M_AXI_WIDTH];
            #pragma HLS ARRAY_PARTITION variable=acc dim=1 complete

            LoopK:
            for (unsigned k = 0; k < sizeK; k++) {
                #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                const unsigned kVecIndex = k / CONFIG_M_AXI_WIDTH;
                CONFIG_DTYPE a_buffer[RowTileSizeD];
                LoopReadA:
                for (unsigned nd = 0; (nd<RowTileSizeD)&&((n*RowTileSizeD+nd)<sizeN); nd++) {
                    #pragma HLS PIPELINE II=1
                    // matrix A is padded on the last dimension but it is accessed by axi-32bits.
                    const unsigned indxS1 = (batch)*sizeN*lastDimPaddedA + (n*RowTileSizeD+nd)*lastDimPaddedA + (k);
                    a_buffer[nd] = A[indxS1];
                }
                LoopM:
                for (unsigned m = 0; m < vecsPerSliceB; m++) {
                    #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                    #pragma HLS PIPELINE II=1
                    const unsigned indxS2 = (batch)*sizeK*vecsPerSliceB + k*vecsPerSliceB + m;
                    const auto b_val = B[indxS2];
                    LoopUnrolled:
                    for (unsigned nd = 0; (nd<RowTileSizeD)&&((n*RowTileSizeD+nd)<sizeN); ++nd) {
                        #pragma HLS UNROLL
                        const auto prev = (k > 0) ? acc[nd][m] : MemoryPackF_t(0.);
                        acc[nd][m] = prev + a_buffer[nd] * b_val;
                        #pragma HLS DEPENDENCE variable=acc inter false
                    }
                }
            }
            LoopWriteD:
            for (unsigned nd = 0; (nd<RowTileSizeD)&&((n*RowTileSizeD+nd)<sizeN); ++nd) {
                LoopWriteM:
                for (unsigned m = 0; m < vecsPerSliceB; ++m) {
                    #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                    #pragma HLS LOOP_FLATTEN
                    #pragma HLS PIPELINE II=1
                    const unsigned indxD = (batch)*sizeN*vecsPerSliceC + (n*RowTileSizeD+nd)*vecsPerSliceC + m;
                    C[indxD] = acc[nd][m];
                }
            }
        }
    }
}

extern "C"{
void task_matmul(
        const CONFIG_DTYPE *inputTn1,
        const MemoryPackF_t *inputTn2,
        MemoryPackF_t *outputTn,
        const unsigned sizeBatch,
        const unsigned sizeN,
        const unsigned sizeK,
        const unsigned sizeM){
#pragma HLS INTERFACE m_axi port=inputTn1 offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi port=inputTn2 offset=slave bundle=gmem2 max_read_burst_length=16 max_write_burst_length=16
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem2 max_read_burst_length=16 max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=inputTn1 bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2 bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn bundle=control
#pragma HLS INTERFACE s_axilite port=sizeBatch bundle=control
#pragma HLS INTERFACE s_axilite port=sizeN bundle=control
#pragma HLS INTERFACE s_axilite port=sizeK bundle=control
#pragma HLS INTERFACE s_axilite port=sizeM bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    MatmulReorderedVectorized_V1(
        inputTn1, inputTn2, outputTn, 
        sizeBatch, sizeN, sizeK, sizeM);

}
}

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
using hlslib::Stream;
using namespace ConfigTaskConcat;

/**
 * @brief      Concatenates the two input tensors over their last dimensions.
 *             Shapes:
 *               inputTn1: BxNxKxD1, inputTn2: BxNxKxD2, outputTn: BxNxKx(D1+D2)
 *             This sub-kernel only handles the input tensors with (D1+D2)<m_axi_width.
 *             This kernel complies with the padded last dim policy:
 *                1) Both of the inputs and output tensors are considered to be padded in the last dim.
 *             The latency will be reported for the input tensors of Shape1=5x1024x1x192, Shape2=5x1024x1x128
 *             This kernel supports burst read/write.
 *             
 * @param[in]  inputTn1  The input tn 1 of rank4
 * @param[in]  inputTn2  The input tn 2 of rank4
 * @param      outputTn  The output tn of rank4
 * @param[in]  dim0      The shape of inputTn1 (dimension 0)
 * @param[in]  dim1      The shape of inputTn1 (dimension 1)
 * @param[in]  dim2      The shape of inputTn1 (dimension 2)
 * @param[in]  dimA3     The shape of inputTn1 (dimension 3)
 * @param[in]  dimB3     The shape of inputTn2 (dimension 3)
 * @param[in]  dimR3     The shape of outputTn (dimension 3)(=dimA3+dimB3)
 */
/*
void ConcatLastDimSubVec_V1(
    const MemoryPackF_t *inputTn1,
    const MemoryPackF_t *inputTn2,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dimA3,
    const unsigned dimB3,
    const unsigned dimR3){

    #pragma HLS INLINE

    assert(dimR3<CONFIG_M_AXI_WIDTH); // sub vec
    constexpr unsigned vecsPerSliceA = 1; // sub vec
    constexpr unsigned vecsPerSliceB = 1; // sub vec
    const unsigned dimR3Padded = MakeDivisible<unsigned>(dimR3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceOutputTn = dimR3Padded / CONFIG_M_AXI_WIDTH;

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            LoopDim2:
            for(unsigned d2=0; d2<dim2; d2++){
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1
                #pragma HLS PIPELINE II=1

                const unsigned indxSA = d0*dim1*dim2*vecsPerSliceA +
                                        d1*dim2*vecsPerSliceA+
                                        d2*vecsPerSliceA+
                                        0;
                const unsigned indxSB = d0*dim1*dim2*vecsPerSliceB +
                                        d1*dim2*vecsPerSliceB+
                                        d2*vecsPerSliceB+
                                        0;
                const unsigned indxD =  d0*dim1*dim2*vecsPerSliceOutputTn +
                                        d1*dim2*vecsPerSliceOutputTn+
                                        d2*vecsPerSliceOutputTn+
                                        0;

                MemoryPackF_t vecA = inputTn1[indxSA];
                MemoryPackF_t vecB = inputTn2[indxSB];
                MemoryPackF_t vecR;
                LoopFillUnrolled:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    vecR[i] = (i<dimA3)?vecA[i]:vecB[i-dimA3];
                }
                outputTn[indxD] = vecR;
            }
        }
    }
}
*/

void ConcatLastDimSubVec_V2(
    const MemoryPackF_t *inputTn1,
    const MemoryPackF_t *inputTn2,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dimA3,
    const unsigned dimB3,
    const unsigned dimR3){

    #pragma HLS INLINE

    assert(dimR3<CONFIG_M_AXI_WIDTH); // sub vec
    constexpr unsigned vecsPerSliceA = 1;
    constexpr unsigned vecsPerSliceB = 1;
    constexpr unsigned dimR3Padded = CONFIG_M_AXI_WIDTH; 
    constexpr unsigned vecsPerSliceOutputTn = 1;

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            LoopDim2:
            for(unsigned d2=0; d2<dim2; d2++){
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1
                #pragma HLS PIPELINE II=1

                const unsigned indxSA = d0*dim1*dim2 +
                                        d1*dim2+
                                        d2;
                const unsigned indxSB = d0*dim1*dim2 +
                                        d1*dim2+
                                        d2;
                const unsigned indxD =  d0*dim1*dim2 +
                                        d1*dim2+
                                        d2;

                MemoryPackF_t vecA = inputTn1[indxSA];
                MemoryPackF_t vecB = inputTn2[indxSB];
                MemoryPackF_t vecR;
                LoopFillUnrolled:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    vecR[i] = (i<dimA3)?vecA[i]:vecB[i-dimA3];
                }
                outputTn[indxD] = vecR;
            }
        }
    }
}

/**
 * @brief      Concatenates the two input tensors over their last dimensions.
 *             Shapes:
 *               inputTn1: BxNxKxD1, inputTn2: BxNxKxD2, outputTn: BxNxKx(D1+D2)
 *             This sub-kernel only handles the input tensors with (D1+D2)>m_axi_width.
 *             Currently, this sub-kernel only supports the inputs with these conditions:
 *                1) D1 > m_axi_width && D1 % m_axi_width=0
 *                2) D2 > m_axi_width && D2 % m_axi_width=0
 *             This kernel complies with the padded last dim policy:
 *                1) Both of the inputs and output tensors are considered to be padded in the last dim.
 *             The latency will be reported for the input tensors of Shape1=5x1024x1x192, Shape2=5x1024x1x128
 *             This kernel supports burst read/write.
 *             
 * @param[in]  inputTn1  The input tn 1 of rank4
 * @param[in]  inputTn2  The input tn 2 of rank4
 * @param      outputTn  The output tn of rank4
 * @param[in]  dim0      The shape of inputTn1 (dimension 0)
 * @param[in]  dim1      The shape of inputTn1 (dimension 1)
 * @param[in]  dim2      The shape of inputTn1 (dimension 2)
 * @param[in]  dimA3     The shape of inputTn1 (dimension 3)
 * @param[in]  dimB3     The shape of inputTn2 (dimension 3)
 * @param[in]  dimR3     The shape of outputTn (dimension 3)(=dimA3+dimB3)
 */
/*
void ConcatLastDimSuperVec_V1(
    const MemoryPackF_t *inputTn1,
    const MemoryPackF_t *inputTn2,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dimA3,
    const unsigned dimB3,
    const unsigned dimR3){

    #pragma HLS INLINE

    assert(dimA3%CONFIG_M_AXI_WIDTH==0);
    assert(dimB3%CONFIG_M_AXI_WIDTH==0);

    const unsigned dimA3Padded = MakeDivisible<unsigned>(dimA3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceA = dimA3Padded / CONFIG_M_AXI_WIDTH;
    const unsigned dimB3Padded = MakeDivisible<unsigned>(dimB3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceB = dimB3Padded / CONFIG_M_AXI_WIDTH;
    const unsigned dimR3Padded = MakeDivisible<unsigned>(dimR3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceOutputTn = dimR3Padded / CONFIG_M_AXI_WIDTH;

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            LoopDim2:
            for(unsigned d2=0; d2<dim2; d2++){
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1
                LoopDimR3_1:
                for(unsigned id3=0; id3<vecsPerSliceA; id3++){
                    #pragma HLS LOOP_TRIPCOUNT min=12 max=12
                    #pragma HLS PIPELINE II=1
                    const unsigned indxS =  d0*dim1*dim2*vecsPerSliceA +
                                            d1*dim2*vecsPerSliceA+
                                            d2*vecsPerSliceA+
                                            id3;
                    const unsigned indxD =  d0*dim1*dim2*vecsPerSliceOutputTn +
                                            d1*dim2*vecsPerSliceOutputTn+
                                            d2*vecsPerSliceOutputTn+
                                            id3;
                    outputTn[indxD] = inputTn1[indxS];
                }
                LoopDimR3_2:
                for(unsigned id3=vecsPerSliceA; id3<vecsPerSliceOutputTn; id3++){
                    #pragma HLS LOOP_TRIPCOUNT min=8 max=8
                    #pragma HLS PIPELINE II=1
                    const unsigned indxS =  d0*dim1*dim2*vecsPerSliceB +
                                            d1*dim2*vecsPerSliceB+
                                            d2*vecsPerSliceB+
                                            (id3-vecsPerSliceA);
                    const unsigned indxD =  d0*dim1*dim2*vecsPerSliceOutputTn +
                                            d1*dim2*vecsPerSliceOutputTn+
                                            d2*vecsPerSliceOutputTn+
                                            id3;
                    outputTn[indxD] = inputTn2[indxS];
                }
            }
        }
    }
}
*/

void ConcatLastDimSuperVec_V2_UnitReadA(
        const MemoryPackF_t *inputTn1,
        Stream<MemoryPackF_t, PipeDepth> &streamOut1,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dimA3,
        const unsigned dimB3,
        const unsigned dimR3){

    assert(dimA3%CONFIG_M_AXI_WIDTH==0);
    assert(dimB3%CONFIG_M_AXI_WIDTH==0);

    const unsigned dimA3Padded = MakeDivisible<unsigned>(dimA3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceA = dimA3Padded / CONFIG_M_AXI_WIDTH;

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            LoopDim2:
            for(unsigned d2=0; d2<dim2; d2++){
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1
                LoopDimA3:
                for(unsigned id3=0; id3<vecsPerSliceA; id3++){
                    #pragma HLS LOOP_TRIPCOUNT min=8 max=8
                    #pragma HLS PIPELINE II=1
                    const unsigned indxS =  d0*dim1*dim2*vecsPerSliceA +
                                            d1*dim2*vecsPerSliceA+
                                            d2*vecsPerSliceA+
                                            id3;
                    streamOut1.Push(inputTn1[indxS]);
                }

            }
        }
    }
}

void ConcatLastDimSuperVec_V2_UnitReadB(
        const MemoryPackF_t *inputTn2,
        Stream<MemoryPackF_t, PipeDepth> &streamOut2,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dimA3,
        const unsigned dimB3,
        const unsigned dimR3){

    assert(dimA3%CONFIG_M_AXI_WIDTH==0);
    assert(dimB3%CONFIG_M_AXI_WIDTH==0);

    const unsigned dimB3Padded = MakeDivisible<unsigned>(dimB3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceB = dimB3Padded / CONFIG_M_AXI_WIDTH;

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            LoopDim2:
            for(unsigned d2=0; d2<dim2; d2++){
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1
                LoopDimB3:
                for(unsigned id3=0; id3<vecsPerSliceB; id3++){
                    #pragma HLS LOOP_TRIPCOUNT min=8 max=8
                    #pragma HLS PIPELINE II=1
                    const unsigned indxS =  d0*dim1*dim2*vecsPerSliceB +
                                            d1*dim2*vecsPerSliceB+
                                            d2*vecsPerSliceB+
                                            id3;
                    streamOut2.Push(inputTn2[indxS]);
                }

            }
        }
    }
}

void ConcatLastDimSuperVec_V2_UnitProcess(
        Stream<MemoryPackF_t, PipeDepth> &streamIn1,
        Stream<MemoryPackF_t, PipeDepth> &streamIn2,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dimA3,
        const unsigned dimB3,
        const unsigned dimR3){

    assert(dimA3%CONFIG_M_AXI_WIDTH==0);
    assert(dimB3%CONFIG_M_AXI_WIDTH==0);

    const unsigned dimA3Padded = MakeDivisible<unsigned>(dimA3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceA = dimA3Padded / CONFIG_M_AXI_WIDTH;
    const unsigned dimB3Padded = MakeDivisible<unsigned>(dimB3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceB = dimB3Padded / CONFIG_M_AXI_WIDTH;
    const unsigned dimR3Padded = MakeDivisible<unsigned>(dimR3, CONFIG_M_AXI_WIDTH); 
    const unsigned vecsPerSliceOutputTn = dimR3Padded / CONFIG_M_AXI_WIDTH;

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            LoopDim2:
            for(unsigned d2=0; d2<dim2; d2++){
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1

                LoopDimR3:
                for(unsigned id3=0; id3<vecsPerSliceOutputTn; id3++){
                    #pragma HLS LOOP_TRIPCOUNT min=12 max=12
                    #pragma HLS PIPELINE II=1

                    const unsigned indxD =  d0*dim1*dim2*vecsPerSliceOutputTn +
                                            d1*dim2*vecsPerSliceOutputTn+
                                            d2*vecsPerSliceOutputTn+
                                            id3;
                    outputTn[indxD] = (id3<vecsPerSliceA) ? streamIn1.Pop() : streamIn2.Pop();
                }

            }
        }
    }
}


void ConcatLastDimSuperVec_V2(
    const MemoryPackF_t *inputTn1,
    const MemoryPackF_t *inputTn2,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dimA3,
    const unsigned dimB3,
    const unsigned dimR3){

#pragma HLS DATAFLOW

    Stream<MemoryPackF_t, PipeDepth> streamA;
#pragma HLS STREAM variable=streamA depth=PipeDepth
    Stream<MemoryPackF_t, PipeDepth> streamB;
#pragma HLS STREAM variable=streamB depth=PipeDepth

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(ConcatLastDimSuperVec_V2_UnitReadA,
        inputTn1, streamA, dim0, dim1, dim2, dimA3, dimB3, dimR3);

    HLSLIB_DATAFLOW_FUNCTION(ConcatLastDimSuperVec_V2_UnitReadB,
        inputTn2, streamB, dim0, dim1, dim2, dimA3, dimB3, dimR3);

    HLSLIB_DATAFLOW_FUNCTION(ConcatLastDimSuperVec_V2_UnitProcess,
        streamA, streamB, outputTn, dim0, dim1, dim2, dimA3, dimB3, dimR3);

    HLSLIB_DATAFLOW_FINALIZE();
}

void ConcatLastDim(
    const MemoryPackF_t *inputTn1,
    const MemoryPackF_t *inputTn2,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dimA3,
    const unsigned dimB3){

#pragma HLS INLINE
    
    const unsigned dimR3 = dimA3 + dimB3;
    if(dimR3<CONFIG_M_AXI_WIDTH){
        ConcatLastDimSubVec_V2(
            inputTn1,
            inputTn2,
            outputTn,
            dim0,
            dim1,
            dim2,
            dimA3,
            dimB3,
            dimR3);
    }else{
        ConcatLastDimSuperVec_V2(
            inputTn1,
            inputTn2,
            outputTn,
            dim0,
            dim1,
            dim2,
            dimA3,
            dimB3,
            dimR3);
    }
}

extern "C" {
void task_concat(
    const MemoryPackF_t *inputTn1,
    const MemoryPackF_t *inputTn2,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dimA3,
    const unsigned dimB3,
    const int concatDim){
#pragma HLS INTERFACE m_axi     port=inputTn1  offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=16
#pragma HLS INTERFACE m_axi     port=inputTn2  offset=slave bundle=gmem2 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi     port=outputTn  offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn1  bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2  bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2      bundle=control
#pragma HLS INTERFACE s_axilite port=dimA3     bundle=control
#pragma HLS INTERFACE s_axilite port=dimB3     bundle=control
#pragma HLS INTERFACE s_axilite port=concatDim bundle=control 
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    if(concatDim==3){
        ConcatLastDim(
            inputTn1,
            inputTn2,
            outputTn,
            dim0,
            dim1,
            dim2,
            dimA3,
            dimB3);
    }

}
}

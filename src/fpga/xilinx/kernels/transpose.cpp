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
using namespace ConfigTaskTranspose;

void BatchTranspose_V6_Burst(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){

    #pragma HLS INLINE

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    assert(PipeDepth1%CONFIG_M_AXI_WIDTH==0);
    assert(PipeDepth2%CONFIG_M_AXI_WIDTH==0);
    assert(dim1%CONFIG_M_AXI_WIDTH==0); // Mandatory

    const unsigned vecsPerDim1 = DivCeil<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerDim2 = DivCeil<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerDepth1 = DivCeil<unsigned>(PipeDepth1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerDepth2 = DivCeil<unsigned>(PipeDepth2, CONFIG_M_AXI_WIDTH);

    CONFIG_DTYPE buff[PipeDepth1][PipeDepth2];
#pragma HLS ARRAY_PARTITION variable=buff cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=buff cyclic factor=8 dim=2

    const unsigned iDim1Bound = DivCeil<unsigned>(dim1, PipeDepth1);

    //const unsigned iDim2Bound = DivCeil<unsigned>(dim2, PipeDepth2);
    const unsigned _dim2 = MakeDivisible<unsigned>(dim2, PipeDepth2);;
    const unsigned iDim2Bound = _dim2 / PipeDepth2;
    const unsigned iDim2BoundSafe = dim2 / PipeDepth2;

    // 1. Burst-only Operations
    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5

        LoopDim2x:
        for(unsigned id2=0; id2<iDim2BoundSafe; id2++){
            #pragma HLS LOOP_TRIPCOUNT min=2 max=2
            LoopDim1x:
            for(unsigned id1=0; id1<iDim1Bound; id1++){
                #pragma HLS LOOP_TRIPCOUNT min=32 max=32

                LoopDim1:
                for(unsigned d1=0; d1<PipeDepth1; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    LoopDim2:
                    for(unsigned d2=0; d2<vecsPerDepth2; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                        #pragma HLS PIPELINE II=1
                        
                        const unsigned indxS = d0*dim1*vecsPerDim2+
                                               (id1*PipeDepth1+d1)*vecsPerDim2+
                                               (id2*vecsPerDepth2+d2);
                        MemoryPackF_t vec = inputTn[indxS];

                        LoopWords:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            buff[d1][d2*CONFIG_M_AXI_WIDTH+i] = vec[i];
                        }
                    }
                }



                LoopX1:
                for(unsigned d1=0; d1<PipeDepth2; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    LoopX2:
                    for(unsigned d2=0; d2<vecsPerDepth1; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                        #pragma HLS PIPELINE II=1

                        const unsigned indxD = d0*dim2*vecsPerDim1+ 
                                               (id2*PipeDepth1+d1)*vecsPerDim1+ 
                                               (id1*vecsPerDepth1+d2);
                        MemoryPackF_t vec;
                        LoopWords2:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            vec[i] = buff[d2*CONFIG_M_AXI_WIDTH+i][d1];
                        }
                        outputTn[indxD] = vec;                  
                    }
                }


            }
        }
    }

    // 2. Non-burst Operations
    if(iDim2Bound!=iDim2BoundSafe){
        const unsigned id2 = iDim2BoundSafe;
        const unsigned vecsPerDepth2Safe = DivCeil<unsigned>(dim2 % PipeDepth2, CONFIG_M_AXI_WIDTH);
        const unsigned PipeDepth2Safe = dim2 % PipeDepth2;

        LoopDim0Rem:
        for(unsigned d0=0; d0<dim0; d0++){
            #pragma HLS LOOP_TRIPCOUNT min=5 max=5

            LoopDim1xRem:
            for(unsigned id1=0; id1<iDim1Bound; id1++){
                #pragma HLS LOOP_TRIPCOUNT min=32 max=32

                LoopDim1Rem:
                for(unsigned d1=0; d1<PipeDepth1; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    LoopDim2Rem:
                    for(unsigned d2=0; d2<vecsPerDepth2Safe; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                        #pragma HLS PIPELINE II=1
                        
                        const unsigned indxS = d0*dim1*vecsPerDim2+
                                               (id1*PipeDepth1+d1)*vecsPerDim2+
                                               (id2*vecsPerDepth2+d2);
                        MemoryPackF_t vec = inputTn[indxS];

                        LoopWordsRem:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            buff[d1][d2*CONFIG_M_AXI_WIDTH+i] = vec[i];
                        }
                    }
                }



                LoopX1Rem:
                for(unsigned d1=0; d1<PipeDepth2Safe; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    LoopX2Rem:
                    for(unsigned d2=0; d2<vecsPerDepth1; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                        #pragma HLS PIPELINE II=1

                        const unsigned indxD = d0*dim2*vecsPerDim1+ 
                                               (id2*PipeDepth1+d1)*vecsPerDim1+ 
                                               (id1*vecsPerDepth1+d2);
                        MemoryPackF_t vec;
                        LoopWords2Rem:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            vec[i] = buff[d2*CONFIG_M_AXI_WIDTH+i][d1];
                        }
                        outputTn[indxD] = vec;                  
                    }
                }
            }
        }
    }
}





    












void BatchTranspose_V5(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){

    #pragma HLS INLINE

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    assert(PipeDepth1%CONFIG_M_AXI_WIDTH==0);
    assert(PipeDepth2%CONFIG_M_AXI_WIDTH==0);
    assert(dim1%CONFIG_M_AXI_WIDTH==0); // Mandatory

    const unsigned vecsPerDim1 = DivCeil<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerDim2 = DivCeil<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerDepth1 = DivCeil<unsigned>(PipeDepth1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerDepth2 = DivCeil<unsigned>(PipeDepth2, CONFIG_M_AXI_WIDTH);


    CONFIG_DTYPE buff[PipeDepth1][PipeDepth2];
#pragma HLS ARRAY_PARTITION variable=buff complete dim=0

    const unsigned iDim1Bound = DivCeil<unsigned>(dim1, PipeDepth1);
    const unsigned iDim2Bound = DivCeil<unsigned>(dim2, PipeDepth2);

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5

        LoopDim2x:
        for(unsigned id2=0; id2<iDim2Bound; id2++){
            #pragma HLS LOOP_TRIPCOUNT min=2 max=2
            LoopDim1x:
            for(unsigned id1=0; id1<iDim1Bound; id1++){
                #pragma HLS LOOP_TRIPCOUNT min=32 max=32

                LoopDim1:
                for(unsigned d1=0; d1<PipeDepth1; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    LoopDim2:
                    for(unsigned d2=0; d2<vecsPerDepth2; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                        #pragma HLS PIPELINE II=1
                        
                        const unsigned indxS = d0*dim1*vecsPerDim2+
                                               (id1*PipeDepth1+d1)*vecsPerDim2+
                                               (id2*vecsPerDepth2+d2);
                        MemoryPackF_t vec = inputTn[indxS];

                        LoopWords:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            buff[d1][d2*CONFIG_M_AXI_WIDTH+i] = vec[i];
                        }
                    }
                }



                LoopX1:
                for(unsigned d1=0; d1<PipeDepth2; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    LoopX2:
                    for(unsigned d2=0; d2<vecsPerDepth1; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                        #pragma HLS PIPELINE II=1

                        const unsigned indxD = d0*dim2*vecsPerDim1+ 
                                               (id2*PipeDepth1+d1)*vecsPerDim1+ 
                                               (id1*vecsPerDepth1+d2);
                        MemoryPackF_t vec;
                        LoopWords2:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            vec[i] = buff[d2*CONFIG_M_AXI_WIDTH+i][d1];
                        }
                        outputTn[indxD] = vec;                  
                    }
                }


            }
        }
    }







}


void BatchTranspose_V4_UnitRead(
    const MemoryPackF_t *inputTn,
    Stream<CONFIG_DTYPE, PipeDepth1> streamWords[PipeDepth2],
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){

    const unsigned vecsPerSlice = DivCeil<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerDepth2 = DivCeil<unsigned>(PipeDepth2, CONFIG_M_AXI_WIDTH);

    const unsigned iDim1Bound = DivCeil<unsigned>(dim1, PipeDepth1);

    const unsigned _dim2 = MakeDivisible<unsigned>(dim2, PipeDepth2);;
    const unsigned iDim2Bound = _dim2 / PipeDepth2;
    const unsigned iDim2BoundSafe = dim2 / PipeDepth2;

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim2x:
        for(unsigned id2=0; id2<iDim2BoundSafe; id2++){
            #pragma HLS LOOP_TRIPCOUNT min=2 max=2
            LoopDim1x:
            for(unsigned id1=0; id1<iDim1Bound; id1++){
                #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                LoopDim1:
                for(unsigned d1=0; d1<PipeDepth1; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    LoopDim2:
                    for(unsigned d2=0; d2<vecsPerDepth2; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                        #pragma HLS PIPELINE II=1
                        const unsigned indxS = d0*dim1*vecsPerSlice+
                                               (id1*PipeDepth1+d1)*vecsPerSlice+
                                               (id2*vecsPerDepth2+d2);
                        MemoryPackF_t vec = inputTn[indxS];
                        LoopWords:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            bool isNotPad = ((id2 * vecsPerDepth2 + d2) * CONFIG_M_AXI_WIDTH + i) < dim2;
                            CONFIG_DTYPE val = vec[i];
                            if (isNotPad) {
                                streamWords[d2 * CONFIG_M_AXI_WIDTH + i].Push(val);
                            }
                        }
                    }
                }
            }
        }
    }

    // Non-burst r/w section
    if(iDim2BoundSafe!=iDim2Bound){
        const unsigned vecsPerDepth2Safe = DivCeil<unsigned>(dim2 % PipeDepth2, CONFIG_M_AXI_WIDTH);
        const unsigned id2 = iDim2BoundSafe;
        LoopDim0Rem:
        for(unsigned d0=0; d0<dim0; d0++){
            #pragma HLS LOOP_TRIPCOUNT min=0 max=0
            LoopDim1xRem:
            for(unsigned id1=0; id1<iDim1Bound; id1++){
                #pragma HLS LOOP_TRIPCOUNT min=0 max=0
                LoopDim1Rem:
                for(unsigned d1=0; d1<PipeDepth1; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=0
                    LoopDim2Rem:
                    for(unsigned d2=0; d2<vecsPerDepth2Safe; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=0 max=0
                        #pragma HLS PIPELINE II=1
                        const unsigned indxS = d0*dim1*vecsPerSlice+
                                               (id1*PipeDepth1+d1)*vecsPerSlice+
                                               (id2*vecsPerDepth2+d2);
                        MemoryPackF_t vec = inputTn[indxS];
                        LoopWordsRem:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            bool isNotPad = ((id2 * vecsPerDepth2 + d2) * CONFIG_M_AXI_WIDTH + i) < dim2;
                            CONFIG_DTYPE val = vec[i];
                            if (isNotPad) {
                                streamWords[d2 * CONFIG_M_AXI_WIDTH + i].Push(val);
                            }
                        }
                    }
                }
            }
        }
    }
}

void BatchTranspose_V4_UnitTranspose(
    Stream<CONFIG_DTYPE, PipeDepth1> streamWords[PipeDepth2],
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){

    const unsigned vecsPerSlice = DivCeil<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerDepth1 = DivCeil<unsigned>(PipeDepth1, CONFIG_M_AXI_WIDTH);

    const unsigned _dim1 = MakeDivisible<unsigned>(dim2, PipeDepth2);;
    const unsigned iDim1Bound = _dim1 / PipeDepth2;
    const unsigned iDim1BoundSafe = dim2 / PipeDepth2;

    const unsigned iDim2Bound = DivCeil<unsigned>(dim1, PipeDepth1);

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim1x:
        for(unsigned id1=0; id1<iDim1BoundSafe; id1++){
            #pragma HLS LOOP_TRIPCOUNT min=2 max=2
            LoopDim2x:
            for(unsigned id2=0; id2<iDim2Bound; id2++){
                #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                LoopDim1:
                for(unsigned d1=0; d1<PipeDepth2; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=32 max=32
                    LoopDim2:
                    for(unsigned d2=0; d2<vecsPerDepth1; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                        #pragma HLS PIPELINE II=1
                        const unsigned indxD = d0*dim2*vecsPerSlice+ 
                                               (id1*PipeDepth2+d1)*vecsPerSlice+ 
                                               (id2*vecsPerDepth1+d2);
                        MemoryPackF_t vec;
                        LoopWords:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            vec[i] = streamWords[d1].Pop();
                        }
                        outputTn[indxD] = vec;
                    }
                }
            }
        }
    }

    // Non-burst r/w section
    if(iDim1Bound!=iDim1BoundSafe){
        const unsigned id1 = iDim1BoundSafe;
        const unsigned PipeDepth2Safe = dim2 % PipeDepth2;
        LoopDim0Rem:
        for(unsigned d0=0; d0<dim0; d0++){
            #pragma HLS LOOP_TRIPCOUNT min=0 max=0
            LoopDim2xRem:
            for(unsigned id2=0; id2<iDim2Bound; id2++){
                #pragma HLS LOOP_TRIPCOUNT min=0 max=0
                LoopDim1Rem:
                for(unsigned d1=0; d1<PipeDepth2Safe; d1++){
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=0
                    LoopDim2Rem:
                    for(unsigned d2=0; d2<vecsPerDepth1; d2++){
                        #pragma HLS LOOP_TRIPCOUNT min=0 max=0
                        #pragma HLS PIPELINE II=1
                        const unsigned indxD = d0*dim2*vecsPerSlice+
                                               (id1*PipeDepth2+d1)*vecsPerSlice+
                                               (id2*vecsPerDepth1+d2);
                        MemoryPackF_t vec;
                        LoopWordsRem:
                        for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                            #pragma HLS UNROLL
                            vec[i] = streamWords[d1].Pop();
                        }
                        outputTn[indxD] = vec;
                    }
                }
            }
        }
    }

}

/**
 * @brief      Calculates transpose of inputTn and writes it in outputTn.
 *             This kernel complies with the padded last dim policy.
 *             The latency is reported for:
 *                  -PipeDepth1=32
 *                  -inputTn of shape 5x1024x64
 *
 * @param[in]  inputTn   The input tn
 * @param      outputTn  The output tn
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  dim2      The dim 2
 */
void BatchTranspose_V4(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){
    
    #pragma HLS INLINE

#pragma HLS DATAFLOW

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    assert(PipeDepth1%CONFIG_M_AXI_WIDTH==0);
    assert(PipeDepth2%CONFIG_M_AXI_WIDTH==0);
    assert(dim1%CONFIG_M_AXI_WIDTH==0); // Mandatory

    Stream<CONFIG_DTYPE, PipeDepth1> streamWords[PipeDepth2];
#pragma HLS STREAM variable=streamWords depth=PipeDepth1

#ifndef HLSLIB_SYNTHESIS
    for (unsigned i = 0; i < PipeDepth2; ++i) {
        streamWords[i].set_name(("streamWords[" + std::to_string(i) + "]").c_str());
    }
#endif

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(BatchTranspose_V4_UnitRead, 
        inputTn, streamWords, dim0, dim1, dim2);
    HLSLIB_DATAFLOW_FUNCTION(BatchTranspose_V4_UnitTranspose, 
        streamWords, outputTn, dim0, dim1, dim2);

    HLSLIB_DATAFLOW_FINALIZE();
}

extern "C"{
void task_transpose(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2){
#pragma HLS INTERFACE m_axi     port=inputTn   offset=slave bundle=gmem1 max_read_burst_length=2 max_write_burst_length=2
#pragma HLS INTERFACE m_axi     port=outputTn  offset=slave bundle=gmem1 max_read_burst_length=2 max_write_burst_length=2
#pragma HLS INTERFACE s_axilite port=inputTn   bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0      bundle=control
#pragma HLS INTERFACE s_axilite port=dim1      bundle=control
#pragma HLS INTERFACE s_axilite port=dim2      bundle=control
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    BatchTranspose_V6_Burst(inputTn, outputTn, dim0, dim1, dim2);
}
}

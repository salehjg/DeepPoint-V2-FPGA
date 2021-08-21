#include <cassert>
#include <iostream>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/DataPack.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using hlslib::Stream;
using namespace ConfigTaskMatOps;

void MatOpsRank4Rankx_V2_UnitRead(
        const MemoryPackF_t *inputTn1,
        const MemoryPackF_t *inputTn2,
        Stream<MemoryPackF_t, PipeDepth> &streamOut1,
        Stream<MemoryPackF_t, PipeDepth> &streamOut2,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3,
        const unsigned dim0B,
        const unsigned dim1B,
        const unsigned dim2B,
        const unsigned dim3B,
        const int rankA,
        const int rankB){
            
    unsigned indxS1, indxS2;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerLastDim = dim3Padded / CONFIG_M_AXI_WIDTH;
    const unsigned dim3BPadded = MakeDivisible<unsigned>(dim3B, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerLastDimB = dim3BPadded / CONFIG_M_AXI_WIDTH;
    MemoryPackF_t sliceBConstant;
    const bool isConstantB = (rankB==1 && dim3B==1);

    assert(rankA>=rankB);
    assert(vecsPerLastDimB<=vecsPerLastDim);
    
    if(isConstantB){
        MemoryPackF_t tmp = inputTn2[0];
        LoopInitConstant:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            sliceBConstant[i] = tmp[0];
        }
    }

    if(isConstantB){
        LoopD0C:
        for(unsigned d0=0; d0<dim0; d0++){
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1

            LoopD1C:
            for(unsigned d1=0; d1<dim1; d1++){
                #pragma HLS LOOP_TRIPCOUNT min=5 max=5

                LoopD2C:
                for(unsigned d2=0; d2<dim2; d2++){
                    #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

                    LoopD3C:
                    for(unsigned iVec3=0; iVec3<vecsPerLastDim; iVec3++){
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                        indxS1= d0*dim1*dim2*vecsPerLastDim+
                                d1*dim2*vecsPerLastDim+
                                d2*vecsPerLastDim+
                                iVec3;

                        streamOut1.Push(inputTn1[indxS1]);
                        streamOut2.Push(sliceBConstant);
                    }
                }
            }
        }
    }else{
        LoopD0:
        for(unsigned d0=0; d0<dim0; d0++){
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1

            LoopD1:
            for(unsigned d1=0; d1<dim1; d1++){
                #pragma HLS LOOP_TRIPCOUNT min=5 max=5

                LoopD2:
                for(unsigned d2=0; d2<dim2; d2++){
                    #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

                    LoopD3:
                    for(unsigned iVec3=0; iVec3<vecsPerLastDim; iVec3++){
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=64 max=64

                        indxS2= (d0*dim1B*dim2B*vecsPerLastDimB*(dim0B==0?0:1)+
                                d1*dim2B*vecsPerLastDimB*(dim1B==0?0:1)+
                                d2*vecsPerLastDimB*(dim2B==0?0:1)+
                                iVec3); 
                        indxS1= d0*dim1*dim2*vecsPerLastDim+
                                d1*dim2*vecsPerLastDim+
                                d2*vecsPerLastDim+
                                iVec3;

                        streamOut1.Push(inputTn1[indxS1]);
                        streamOut2.Push(inputTn2[indxS2]);
                    }
                }
            }
        }
    }       
}

void MatOpsRank4Rankx_V2_UnitProcess(
        Stream<MemoryPackF_t, PipeDepth> &streamIn1,
        Stream<MemoryPackF_t, PipeDepth> &streamIn2,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3,
        const unsigned dim0B,
        const unsigned dim1B,
        const unsigned dim2B,
        const unsigned dim3B,
        const int rankA,
        const int rankB, 
        const int mode){
            
    unsigned indxS1;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerLastDim = dim3Padded / CONFIG_M_AXI_WIDTH;
    const unsigned dim3BPadded = MakeDivisible<unsigned>(dim3B, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerLastDimB = dim3BPadded / CONFIG_M_AXI_WIDTH;
    MemoryPackF_t sliceBConstant;
    const bool isConstantB = (rankB==1 && dim3B==1);

    assert(rankA>=rankB);
    assert(vecsPerLastDimB<=vecsPerLastDim);

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    
    LoopD0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1

        LoopD1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=5 max=5

            LoopD2:
            for(unsigned d2=0; d2<dim2; d2++){
                #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

                LoopD3:
                for(unsigned iVec3=0; iVec3<vecsPerLastDim; iVec3++){
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=64 max=64 

                    MemoryPackF_t sliceA = streamIn1.Pop();
                    MemoryPackF_t sliceB = streamIn2.Pop();
                    MemoryPackF_t sliceO;

                    if(mode==0){
                        // Add
                        sliceO = sliceA + sliceB;
                    }
                    else if(mode==1){
                        // Sub
                        sliceO = sliceA - sliceB;
                    }
                    else if(mode==2){
                        // Mul (element wise)
                        sliceO = sliceA * sliceB;
                    }
                    else if(mode==3){
                        // Div (element wise)
                        sliceO = sliceA / sliceB;
                    }

                    indxS1= d0*dim1*dim2*vecsPerLastDim+
                            d1*dim2*vecsPerLastDim+
                            d2*vecsPerLastDim+
                            iVec3;
                    outputTn[indxS1] = sliceO;
                }
            }
        }
    }   
}


/**
 * @brief      Performs Addition, Subtraction, Multiplication, and Division on two tensors.
 *             The first tensor could be of rank r1 where 1<=r1<=4
 *             The second tensor should be of rank r2 where 1<=r2<=r1
 *             This kernel complies with the padded last dim policy.
 *             The shape of the first tensor should be aligned by the last dimension like:
 *               rank=2 tensor 2x3: dim0=1, dim1=1, dim2=2, dim3=3
 *             The shape of the second tensor should also be aligned by the last dimension like:
 *               rank=2 tensor 2x3: dim0B=0, dim1B=0, dim2B=2, dim3B=3
 *             The latency will be reported for an input tensor of shape 5x1024x1024.
 *             This kernel supports burst read/write.
 *
 * @param[in]  inputTn1  The input tn 1
 * @param[in]  inputTn2  The input tn 2
 * @param      outputTn  The output tn of the same shape as inputTn1
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  dim2      The dim 2
 * @param[in]  dim3      The dim 3
 * @param[in]  dim0B     The dim 0 b
 * @param[in]  dim1B     The dim 1 b
 * @param[in]  dim2B     The dim 2 b
 * @param[in]  dim3B     The dim 3 b
 * @param[in]  rankA     The rank a
 * @param[in]  rankB     The rank b
 * @param[in]  mode      The mode
 */

void MatOpsRank4Rankx_V2(
        const MemoryPackF_t *inputTn1, //is always of rank4 (forced)
        const MemoryPackF_t *inputTn2, //rank4 or less
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3,
        const unsigned dim0B,
        const unsigned dim1B,
        const unsigned dim2B,
        const unsigned dim3B,
        const int rankA,
        const int rankB, 
        const int mode){

#pragma HLS DATAFLOW

    Stream<MemoryPackF_t, PipeDepth> stream1;
#pragma HLS STREAM variable=stream1 depth=PipeDepth
    Stream<MemoryPackF_t, PipeDepth> stream2;
#pragma HLS STREAM variable=stream2 depth=PipeDepth

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(MatOpsRank4Rankx_V2_UnitRead,
    		inputTn1, inputTn2, stream1, stream2, dim0, dim1, dim2, dim3, dim0B, dim1B, dim2B, dim3B, rankA, rankB);
    HLSLIB_DATAFLOW_FUNCTION(MatOpsRank4Rankx_V2_UnitProcess, 
        stream1, stream2, outputTn, dim0, dim1, dim2, dim3, dim0B, dim1B, dim2B, dim3B, rankA, rankB, mode);

    HLSLIB_DATAFLOW_FINALIZE();
}

/**
 * @brief      Performs Addition, Subtraction, Multiplication, and Division on two tensors.
 *             The first tensor could be of rank r1 where 1<=r1<=4
 *             The second tensor should be of rank r2 where 1<=r2<=r1
 *             This kernel complies with the padded last dim policy.
 *             The shape of the first tensor should be aligned by the last dimension like:
 *               rank=2 tensor 2x3: dim0=1, dim1=1, dim2=2, dim3=3
 *             The shape of the second tensor should also be aligned by the last dimension like:
 *               rank=2 tensor 2x3: dim0B=0, dim1B=0, dim2B=2, dim3B=3
 *             This kernel DOES NOT supports burst read/write.
 *
 * @param[in]  inputTn1  The input tn 1
 * @param[in]  inputTn2  The input tn 2
 * @param      outputTn  The output tn of the same shape as inputTn1
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  dim2      The dim 2
 * @param[in]  dim3      The dim 3
 * @param[in]  dim0B     The dim 0 b
 * @param[in]  dim1B     The dim 1 b
 * @param[in]  dim2B     The dim 2 b
 * @param[in]  dim3B     The dim 3 b
 * @param[in]  rankA     The rank a
 * @param[in]  rankB     The rank b
 * @param[in]  mode      The mode
 */
/*
void MatOpsRank4Rankx(
        const MemoryPackF_t *inputTn1, //is always of rank4 (forced)
        const MemoryPackF_t *inputTn2, //rank4 or less
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3,
        const unsigned dim0B,
        const unsigned dim1B,
        const unsigned dim2B,
        const unsigned dim3B, //should be 1 for a constant
        const int rankA,
        const int rankB, //should be 1 for a constant
        const int mode){
    
    #pragma HLS INLINE
    
    unsigned indxS1,indxS2;
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerLastDim = dim3Padded / CONFIG_M_AXI_WIDTH;
    const unsigned dim3BPadded = MakeDivisible<unsigned>(dim3B, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerLastDimB = dim3BPadded / CONFIG_M_AXI_WIDTH;
    MemoryPackF_t sliceBConstant;
    const bool isConstantB = (rankB==1 && dim3B==1);

    assert(rankA>=rankB);
    assert(vecsPerLastDimB<=vecsPerLastDim);

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    
    if(isConstantB){
        MemoryPackF_t tmp = inputTn2[0];
        LoopInitConstant:
        for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
            sliceBConstant[i] = tmp[0];
        }
    }

    LoopD0:
    for(unsigned d0=0; d0<dim0; d0++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1

        LoopD1:
        for(unsigned d1=0; d1<dim1; d1++){
            #pragma HLS LOOP_TRIPCOUNT min=5 max=5

            LoopD2:
            for(unsigned d2=0; d2<dim2; d2++){
                #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

                LoopD3:
                for(unsigned iVec3=0; iVec3<vecsPerLastDim; iVec3++){
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=64 max=64

                    indxS1= d0*dim1*dim2*vecsPerLastDim+
                            d1*dim2*vecsPerLastDim+
                            d2*vecsPerLastDim+
                            iVec3;
                    indxS2= isConstantB? // to avoid unwanted outofbound mem access on hw/emu.
                            0:
                            (d0*dim1B*dim2B*vecsPerLastDimB*(dim0B==0?0:1)+
                            d1*dim2B*vecsPerLastDimB*(dim1B==0?0:1)+
                            d2*vecsPerLastDimB*(dim2B==0?0:1)+
                            iVec3); 

                    MemoryPackF_t sliceB;
                    if(isConstantB){
                        sliceB = sliceBConstant;
                    }else{
                        sliceB = inputTn2[indxS2];
                    }

                    if(mode==0){
                        // Add
                        outputTn[indxS1] = inputTn1[indxS1] + sliceB;
                    }
                    else if(mode==1){
                        // Sub
                        outputTn[indxS1] = inputTn1[indxS1] - sliceB;
                    }
                    else if(mode==2){
                        // Mul (element wise)
                        outputTn[indxS1] = inputTn1[indxS1] * sliceB;
                    }
                    else if(mode==3){
                        // Div (element wise)
                        outputTn[indxS1] = inputTn1[indxS1] / sliceB;
                    }
                }
            }
        }
    }
}
*/

extern "C" {
void task_matops(
        const MemoryPackF_t *inputTn1, //is always of rank4 (forced)
        const MemoryPackF_t *inputTn2, //rank4 or less
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3,
        const unsigned dim0B,
        const unsigned dim1B,
        const unsigned dim2B,
        const unsigned dim3B, 
        const int rankA,
        const int rankB,
        const int mode){

#pragma HLS INTERFACE m_axi port=inputTn1 offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=16
#pragma HLS INTERFACE m_axi port=inputTn2 offset=slave bundle=gmem2 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn1 bundle=control
#pragma HLS INTERFACE s_axilite port=inputTn2 bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=dim2 bundle=control
#pragma HLS INTERFACE s_axilite port=dim3 bundle=control
#pragma HLS INTERFACE s_axilite port=dim0B bundle=control
#pragma HLS INTERFACE s_axilite port=dim1B bundle=control
#pragma HLS INTERFACE s_axilite port=dim2B bundle=control
#pragma HLS INTERFACE s_axilite port=dim3B bundle=control
#pragma HLS INTERFACE s_axilite port=rankA bundle=control
#pragma HLS INTERFACE s_axilite port=rankB bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    MatOpsRank4Rankx_V2(
        inputTn1,
        inputTn2,
        outputTn,
        dim0,
        dim1,
        dim2,
        dim3,
        dim0B,
        dim1B,
        dim2B,
        dim3B,
        rankA,
        rankB,
        mode);

}
}

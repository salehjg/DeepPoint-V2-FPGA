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
using namespace ConfigTaskGather;

//The latency for inputTn of shape 5x1024x64 and indicesTn of shape 5x1024x20
void UnitReadIndices(
    const unsigned *indicesTn,
    Stream<unsigned, PipeDepth> &streamIndices,
    unsigned indicesDim0,
    unsigned indicesDim1,
    unsigned indicesDim2){

    const unsigned B = indicesDim0;
    const unsigned N = indicesDim1;
    const unsigned K = indicesDim2;

    // IndicesTn: BxNxK
    // InputTn:   BxNxD
    
    const unsigned paddedK = MakeDivisible<unsigned>(K, CONFIG_M_AXI_WIDTH); 

    LoopB:
    for(unsigned b=0; b<B; b++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopN:
        for(unsigned n=0; n<N; n++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            LoopReadIndices:
            for(unsigned k=0; k<K; k++){
                #pragma HLS LOOP_TRIPCOUNT min=20 max=20
                #pragma HLS PIPELINE II=1   
                const unsigned indxI = b*N*paddedK + n*paddedK + k;
                unsigned val = indicesTn[indxI];
                streamIndices.Push(val);
            }
        }
    }
}

//The latency for inputTn of shape 5x1024x64 and indicesTn of shape 5x1024x20
void UnitGather(
    Stream<unsigned, PipeDepth> &streamIndices,
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    unsigned inputDim0,
    unsigned inputDim1,
    unsigned inputDim2,
    unsigned indicesDim0,
    unsigned indicesDim1,
    unsigned indicesDim2){

    assert(inputDim0 == indicesDim0); //B
    assert(inputDim1 == indicesDim1); //N
    const unsigned B = inputDim0;
    const unsigned N = inputDim1;
    const unsigned K = indicesDim2;
    const unsigned D = inputDim2;

    // IndicesTn: BxNxK
    // InputTn:   BxNxD
    
    const unsigned paddedK = MakeDivisible<unsigned>(K, CONFIG_M_AXI_WIDTH);
    const unsigned paddedD = MakeDivisible<unsigned>(D, CONFIG_M_AXI_WIDTH); 

    const unsigned vecsPerSliceIndicesTn = paddedK / CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceInputTn = paddedD / CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceOutputTn = vecsPerSliceInputTn;

    unsigned currentLocalIndex;

    LoopB:
    for(unsigned b=0; b<B; b++){
        #pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopN:
        for(unsigned n=0; n<N; n++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024

            LoopK:
            for(unsigned k=0; k<K; k++){
                #pragma HLS LOOP_TRIPCOUNT min=20 max=20

                // Copy data slices related to 'currentLocalIndex'
                LoopD:
                for(unsigned id=0; id<vecsPerSliceInputTn; id++){
                    #pragma HLS LOOP_TRIPCOUNT min=4 max=4
                    #pragma HLS PIPELINE II=1

                	if(id==0){
                		currentLocalIndex = streamIndices.Pop();
                	}
                    const unsigned indxS =  b*N*vecsPerSliceInputTn + 
                                            currentLocalIndex*vecsPerSliceInputTn +
                                            id;
                    const unsigned indxD =  b*N*K*vecsPerSliceOutputTn + 
                                            n*K*vecsPerSliceInputTn +
                                            k*vecsPerSliceInputTn +
                                            id;
                    outputTn[indxD] = inputTn[indxS];
                }
            }
        }
    }

}



/**
 * @brief      Gathers elements of inputTn with the indices provided in 
 *               indicesTn in the given axis.
 *             The latency will be reported for inputTn of shape 5x1024x64 and indicesTn of shape 5x1024x20.
 *             This kernel complies with the padded last dim policy:
 *               1) inputTn and outputTn should be padded in the last dim, so it would be 
 *                  divisible by m_axi512's width. 
 *               2) indicesTn uses 32-bit axi but nevertheless it should be padded in 
 *                  the last dim to be divisible by m_axi512's width.
 * @param[in]  inputTn      The input tn
 * @param[in]  indicesTn    The indices tn
 * @param      outputTn     The output tn
 * @param[in]  inputDim0    The input dim 0
 * @param[in]  inputDim1    The input dim 1
 * @param[in]  inputDim2    The input dim 2
 * @param[in]  indicesDim0  The indices dim 0
 * @param[in]  indicesDim1  The indices dim 1
 * @param[in]  indicesDim2  The indices dim 2
 */
void GatherAxis1_V2(
    const MemoryPackF_t *inputTn,
    const unsigned *indicesTn,
    MemoryPackF_t *outputTn,
    unsigned inputDim0,
    unsigned inputDim1,
    unsigned inputDim2,
    unsigned indicesDim0,
    unsigned indicesDim1,
    unsigned indicesDim2){

    // DO NOT use inline pragma for a sub-function with dataflow pragma, as it might cause the kernel to get stuck.
    //#pragma HLS INLINE

#pragma HLS DATAFLOW
    
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    Stream<unsigned, PipeDepth> streamIndices;
#pragma HLS STREAM variable=streamIndices depth=PipeDepth
#ifndef HLSLIB_SYNTHESIS
    streamIndices.set_name("streamIndices");
#endif
    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(UnitReadIndices, 
        indicesTn, streamIndices, 
        indicesDim0, indicesDim1, indicesDim2);
    HLSLIB_DATAFLOW_FUNCTION(UnitGather, 
        streamIndices, inputTn, outputTn, 
        inputDim0, inputDim1, inputDim2, 
        indicesDim0, indicesDim1, indicesDim2);

    HLSLIB_DATAFLOW_FINALIZE();
}

extern "C"{
void task_gather(
    const MemoryPackF_t *inputTn,
    const unsigned *indicesTn,
    MemoryPackF_t *outputTn,
    unsigned indicesAxis,
    unsigned inputDim0,
    unsigned inputDim1,
    unsigned inputDim2,
    unsigned indicesDim0,
    unsigned indicesDim1,
    unsigned indicesDim2){
#pragma HLS INTERFACE m_axi     port=inputTn        offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi     port=indicesTn      offset=slave bundle=gmem2 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi     port=outputTn       offset=slave bundle=gmem3 max_read_burst_length=2 max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=inputTn        bundle=control
#pragma HLS INTERFACE s_axilite port=indicesTn      bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn       bundle=control
#pragma HLS INTERFACE s_axilite port=indicesAxis    bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim0      bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim1      bundle=control
#pragma HLS INTERFACE s_axilite port=inputDim2      bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim0    bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim1    bundle=control
#pragma HLS INTERFACE s_axilite port=indicesDim2    bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    if(indicesAxis==1){
        GatherAxis1_V2(
            inputTn,
            indicesTn,
            outputTn,
            inputDim0,
            inputDim1,
            inputDim2,
            indicesDim0,
            indicesDim1,
            indicesDim2);
    }
    
}
}

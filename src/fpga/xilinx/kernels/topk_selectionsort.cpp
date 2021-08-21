#include <cassert>
#include <iostream>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskTopK;
using hlslib::Stream;

void UnitReadInput(
    const MemoryPackF_t *inputTn,
    Stream<MemoryPackF_t, PipeDepth> &streamInputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice){

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(dim1%CONFIG_M_AXI_WIDTH==0);

    unsigned indxS;

    For_Main: 
    for(unsigned batch=0; batch<dim0; batch+=UnitCount){
        LoopVecsPerPE:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            LoopPEs:
            for(unsigned iPE=0; iPE<UnitCount; iPE++){
                indxS = (batch+iPE)*vecsPerSlice + iVec;
                streamInputTn.Push(inputTn[indxS]);
            }
        }
    }

}

void UnitWriteOutput(
    MemoryPackI_t *indicesSplitedTn,
    Stream<MemoryPackI_t, PipeDepth> &streamIndices,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerOutputSlice){

    // FIFO depth should be greater or equal to 'vecsPerOutputSlice' for PE's not to stall.
    assert(PipeDepth>=vecsPerOutputSlice);

    unsigned indxD;

    For_Main: 
    for(unsigned batch=0; batch<dim0; batch+=UnitCount){
        LoopPEs:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE:
            for(unsigned iVec=0; iVec<vecsPerOutputSlice; iVec++){
                indxD = (batch+iPE)*vecsPerOutputSlice + iVec;
                indicesSplitedTn[indxD] = streamIndices.Pop();
            }
        }
    }
}

//latency reported for [5x1024]x1024, k=20, unitcount=8, m_axi_width=16, pipe_depth=2
void UnitProcessingElement(
    Stream<MemoryPackF_t, PipeDepth> &streamDataIn,
    Stream<MemoryPackF_t, PipeDepth> &streamDataOut,
    Stream<MemoryPackI_t, PipeDepth> &streamIndicesIn,
    Stream<MemoryPackI_t, PipeDepth> &streamIndicesOut,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice,
    const unsigned vecsPerOutputSlice,
    const unsigned kValue,
    const unsigned unitIndex){
    
    // Only divisible 'dim1' by maxi width is supported so far.
    assert(dim1%CONFIG_M_AXI_WIDTH==0);

    // Length of the PE's local buffers('MaxSliceLen') should be greater or equal to 'dim1'.
    assert(dim1<=MaxSliceLen);

    // FIFO depth should be greater or equal to 'vecsPerOutputSlice' for PE's not to stall.
    assert(PipeDepth>=vecsPerOutputSlice);

    unsigned min_idx;
    unsigned indxS,indxD;

    CONFIG_DTYPE sliceData[MaxSliceLen];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=sliceData cyclic factor=CONFIG_M_AXI_WIDTH dim=1)
    //DO_PRAGMA(HLS ARRAY_PARTITION variable=sliceData complete dim=1)

    unsigned sliceIndices[MaxSliceLen];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=sliceIndices cyclic factor=CONFIG_M_AXI_WIDTH dim=1)
    //DO_PRAGMA(HLS ARRAY_PARTITION variable=sliceIndices complete dim=1)

    MemoryPackF_t sliceSubVec;
    MemoryPackI_t outputCache;
    unsigned outputCacheVecSubIdx;

    LoopMain: for(unsigned batch=0; batch<dim0; batch+=UnitCount){
#pragma HLS LOOP_TRIPCOUNT min=640 max=640
        ///TODO: Check for out of bound batch index 
        //--------------------------------------------------
        // 1. Read current slice and indices into local memory.
        LoopReadSlice: for(unsigned idx=0; idx<vecsPerSlice; idx++){
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64
            
            LoopInputPass01:
            for(unsigned iPE=0; iPE<UnitCount-unitIndex; iPE++){
                #pragma HLS PIPELINE II=1
                MemoryPackF_t vec = streamDataIn.Pop();
                if(iPE>0){
                    // Pass the data to other PEs and just keep the last one for this PE
                    if(unitIndex<(UnitCount-1)){
                        streamDataOut.Push(vec);
                    }
                }else{
                    sliceSubVec = vec;
                }
            }
            
            const unsigned offsetLocal = idx*CONFIG_M_AXI_WIDTH;
            LoopReadUnroll1:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                const unsigned indxLocal = offsetLocal+i; 
                sliceData[indxLocal] = sliceSubVec[i];
                sliceIndices[indxLocal] = indxLocal;
            }
        }

        //--------------------------------------------------
        // 2. Run sorting algorithm on the local memory.
        const unsigned _len = ( kValue*( (dim1-1) + (dim1-kValue) ) )/2;
        unsigned i,j; i = 0; j = 1;
        min_idx = 0;

        LoopFusedSort0:
        for(unsigned iter=0; iter<_len;iter++){
            #pragma HLS PIPELINE II=1
			#pragma HLS LOOP_TRIPCOUNT min=20270 max=20270

            if(sliceData[j] < sliceData[min_idx]){
                min_idx = j;
            }

            //------------------------------
            //Fused loop's house keeping stuff
            if(j==dim1-1){
                //if(min_idx != i)
                {
                    //Commented lines are to avoid unnecessary memory accesses.
                    //They don't affect the REQUIRED output of this PE.

                    //float tmp = sliceData[min_idx];
                    sliceData[min_idx] = sliceData[i];
                    //sliceData[i] = tmp;
                    //--------------------------------
                    unsigned tmp2 = sliceIndices[min_idx];
                    sliceIndices[min_idx] = sliceIndices[i];
                    sliceIndices[i] = tmp2;

                    //--------------------------------
                    outputCacheVecSubIdx = i%CONFIG_M_AXI_WIDTH;
                    outputCache[outputCacheVecSubIdx] = tmp2;
                    if(outputCacheVecSubIdx==(CONFIG_M_AXI_WIDTH-1) || i==(kValue-1) ){
                        streamIndicesOut.Push(outputCache);
#ifdef KERNEL_LOGS
                        cout<<"PE"<<unitIndex<<": "<<" Sorted Vec i="<<i<<endl;
#endif
                    }
                }
                //--------------------------
                i++;
                j=i+1;
                //--------------------------
                min_idx = i;
            }else{
                j++;
            }
        }

        //--------------------------------------------------
        
        // 3. Handle incoming data of streamIndicesIn from other PEs.
        const unsigned _len2 = UnitCount-unitIndex-1;
        LoopHandleOtherPEsOutput:
        for(unsigned iPE=0; iPE<_len2; iPE++){
			#pragma HLS LOOP_TRIPCOUNT min=8 max=8
            ForOutputVecsPerPEs:
            for(unsigned iVec=0; iVec<vecsPerOutputSlice; iVec++){
                if(unitIndex<(UnitCount-1)){
                    streamIndicesOut.Push(streamIndicesIn.Pop());
                }
#ifdef KERNEL_LOGS
                cout<<"*PE"<<unitIndex<<": "<<"Handling Other PE Results, "<<"Pop'ed streamIndicesIn"<<endl;
#endif
            }
        }
    }
#ifdef KERNEL_LOGS
    cout<<"==PE"<<unitIndex<<": "<<"FINISHED"<<endl;
#endif
}

extern "C"{
void task_topk(
        const MemoryPackF_t *inputTn,
        MemoryPackI_t *indicesSplitedTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned kValue,
        const unsigned vecsPerSlice,
        const unsigned vecsPerOutputSlice){

#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=indicesSplitedTn offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=indicesSplitedTn bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=kValue bundle=control
#pragma HLS INTERFACE s_axilite port=vecsPerSlice bundle=control
#pragma HLS INTERFACE s_axilite port=vecsPerOutputSlice bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    Stream<MemoryPackF_t, PipeDepth> streamsData[UnitCount+1];
#pragma HLS STREAM variable=streamsData depth=PipeDepth

    Stream<MemoryPackI_t, PipeDepth> streamsIndices[UnitCount+1];
#pragma HLS STREAM variable=streamsIndices depth=PipeDepth

#ifndef HLSLIB_SYNTHESIS
    // Name the arrays of channels for debugging purposes
    for (unsigned i = 0; i < UnitCount+1; i++) {
        streamsData[i].set_name(("streamsData[" + std::to_string(i) + "]").c_str());
    }
    for (unsigned n = 0; n < UnitCount+1; n++) {
        streamsIndices[n].set_name(("streamsIndices[" + std::to_string(n) + "]").c_str());
    }
#endif
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(UnitReadInput, inputTn, streamsData[0], dim0, dim1, vecsPerSlice);
    
    for (unsigned iPE = 0; iPE < UnitCount; iPE++) {
#pragma HLS UNROLL
        HLSLIB_DATAFLOW_FUNCTION(UnitProcessingElement,
            streamsData[iPE],
            streamsData[iPE+1],
            streamsIndices[iPE+1],
            streamsIndices[iPE],
            dim0,
            dim1,
            vecsPerSlice,
            vecsPerOutputSlice,
            kValue,
            iPE);
    }
    
    HLSLIB_DATAFLOW_FUNCTION(UnitWriteOutput, indicesSplitedTn, streamsIndices[0], dim0, dim1, vecsPerOutputSlice);

    HLSLIB_DATAFLOW_FINALIZE();
    

}
}

#include <cassert>
#include <iostream>
#include <limits>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using namespace ConfigTaskTopK;
using hlslib::Stream;

constexpr unsigned CUCount = 16;

void UnitReadInput(
    const MemoryPackF_t *inputTn,
    Stream<CONFIG_DTYPE, PipeDepth> &streamData,
    Stream<unsigned, PipeDepth> &streamIndices,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice){

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(dim1%CONFIG_M_AXI_WIDTH==0);

    unsigned indxS;

    For_Main: 
    for(unsigned batch=0; batch<dim0; batch+=1){
        LoopVecsPerPE:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            indxS = (batch)*vecsPerSlice + iVec;
            MemoryPackF_t vec = inputTn[indxS];
            //cout<<"UnitReadInput: Valid, batch="<<batch<<", iVec="<<iVec<<endl;
            LoopReadVec0:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                streamData.Push(vec[i]);
                const unsigned sliceSubIndex = iVec*CONFIG_M_AXI_WIDTH+i;
                streamIndices.Push(sliceSubIndex);
            }
        }

        LoopPushInvalidDataIn:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            //cout<<"UnitReadInput: Invalid, iVec="<<iVec<<endl;
            LoopReadVec1:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                streamData.Push(std::numeric_limits<float>::max());
                streamIndices.Push(0);
            }
        }
    }

    cout<<"UnitReadInput: FINISHED"<<endl;
}

void UnitWriteOutput(
    MemoryPackI_t *indicesSplitedTn,
    Stream<CONFIG_DTYPE, PipeDepth> &streamData,
    Stream<unsigned, PipeDepth> &streamIndices,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice,
    const unsigned vecsPerOutputSlice){

    unsigned indxD;

    For_Main: 
    for(unsigned batch=0; batch<dim0; batch+=1){
        LoopFlushInvalidOutputs:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            LoopGetSortedIndices0:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                streamIndices.Pop();
                streamData.Pop();
            }
        }
        LoopVecsPerPE:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            indxD = (batch)*vecsPerOutputSlice + iVec;
            MemoryPackI_t vec;
            //cout<<"UnitWriteOutput: Valid, batch="<<batch<<", iVec="<<iVec<<endl;

            LoopGetSortedIndices1:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                vec[i] = streamIndices.Pop();
                streamData.Pop();
            }

            if(iVec<vecsPerOutputSlice){
                indicesSplitedTn[indxD] = vec;

                cout<<"UnitWriteOutput: Result[iVec:"<<iVec<<"]=";
                LoopDebug:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    cout<<vec[i]<<",";
                }
                cout<<"\n"<<std::flush;
            }

        } 
    }

    cout<<"UnitWriteOutput: FINISHED"<<endl;
}

void InsertionSortCell(
    Stream<CONFIG_DTYPE, PipeDepth> &inData, 
    Stream<CONFIG_DTYPE, PipeDepth> &outData,
    Stream<unsigned, PipeDepth> &inIndex, 
    Stream<unsigned, PipeDepth> &outIndex,
    const unsigned dim0){
    CONFIG_DTYPE localData = std::numeric_limits<float>::min();
    CONFIG_DTYPE localIndex = 0;
    LoopBatch:
    for(unsigned batch=0; batch<2*dim0; batch++){
        LoopArrayLen:
        for(unsigned d1=0; d1<CUCount; d1++){
            if(batch%2==0 && d1==0){
                localData=std::numeric_limits<float>::min();
                localIndex=0;
            }

            CONFIG_DTYPE inDataCopy = inData.Pop();
            unsigned inIndexCopy = inIndex.Pop();
            //cout<<"II:"<<inDataCopy<<endl<<std::flush;
            if(inDataCopy > localData) {
                outData.Push(localData);
                localData = inDataCopy;
                outIndex.Push(localIndex);
                localIndex = inIndexCopy;
            }else{
                outData.Push(inDataCopy);
                outIndex.Push(inIndexCopy);
            }
        }
        //cout<<"Cell: batch="<<batch<<endl<<std::flush;;
    }
    //cout<<"Cell: FINISHED"<<endl;
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

#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1 max_read_burst_length=64 max_write_burst_length=2
#pragma HLS INTERFACE m_axi port=indicesSplitedTn offset=slave bundle=gmem2 max_read_burst_length=2 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=indicesSplitedTn bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=kValue bundle=control
#pragma HLS INTERFACE s_axilite port=vecsPerSlice bundle=control
#pragma HLS INTERFACE s_axilite port=vecsPerOutputSlice bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    Stream<CONFIG_DTYPE, PipeDepth> streamsData[CUCount+1];
#pragma HLS STREAM variable=streamsData depth=PipeDepth

    Stream<unsigned, PipeDepth> streamsIndices[CUCount+1];
#pragma HLS STREAM variable=streamsIndices depth=PipeDepth

#ifndef HLSLIB_SYNTHESIS
    // Name the arrays of channels for debugging purposes
    for (unsigned i = 0; i < CUCount+1; i++) {
        streamsData[i].set_name(("streamsData[" + std::to_string(i) + "]").c_str());
    }
    for (unsigned n = 0; n < CUCount+1; n++) {
        streamsIndices[n].set_name(("streamsIndices[" + std::to_string(n) + "]").c_str());
    }
#endif
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(UnitReadInput, inputTn, streamsData[0], streamsIndices[0], dim0, dim1, vecsPerSlice);
    
    for (unsigned iPE = 0; iPE < CUCount; iPE++) {
#pragma HLS UNROLL
        HLSLIB_DATAFLOW_FUNCTION(InsertionSortCell,
            streamsData[iPE],
            streamsData[iPE+1],
            streamsIndices[iPE],
            streamsIndices[iPE+1],
            dim0);
    }
    
    HLSLIB_DATAFLOW_FUNCTION(UnitWriteOutput, indicesSplitedTn, streamsData[CUCount], streamsIndices[CUCount], dim0, dim1, vecsPerSlice, vecsPerOutputSlice);

    HLSLIB_DATAFLOW_FINALIZE();
}
}

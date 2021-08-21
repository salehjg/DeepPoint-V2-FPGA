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

/**
 * @brief      2-way merge sorting algorithm with additional logic to provide indices. 
 *             The code is adapted from the open-source book "Parallel Programming for FPGAs" with minor changes.
 *
 * @param      inLocalBuff     pointer to a local buffer that contains slice data to be sorted in place.
 * @param      inLocalIndices  pointer to a local buffer that indices of sorted data will be stored in.
 */
void MergeSortWithIndices(
    CONFIG_DTYPE *inLocalBuff,
    unsigned *inLocalIndices){
#pragma HLS INLINE

    CONFIG_DTYPE temp[MaxSliceLen];
#pragma HLS RESOURCE variable=temp core=RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable=temp cyclic factor=CONFIG_M_AXI_WIDTH dim=1

    ///TODO: Try implementing the algorithm without the need for indicesTemp.
    unsigned indicesTemp[MaxSliceLen];
#pragma HLS ARRAY_PARTITION variable=indicesTemp cyclic factor=CONFIG_M_AXI_WIDTH dim=1
//#pragma HLS RESOURCE variable=indicesTemp core=RAM_2P_URAM

    LoopStage:
    for (int width = 1; width < MaxSliceLen; width = 2 * width) {
        int f1 = 0;
        int f2 = width;
        int i2 = width;
        int i3 = 2*width;
        if(i2 >= MaxSliceLen) i2 = MaxSliceLen;
        if(i3 >= MaxSliceLen) i3 = MaxSliceLen;

        LoopMergeArrays:
        for (int i = 0; i < MaxSliceLen; i++) {
            #pragma HLS pipeline II=1
            ///TODO: Try to fix the II-violation(currently II=2)
            ///      II=1 is achievable for integer data types.

            CONFIG_DTYPE t1 = inLocalBuff[f1];
            CONFIG_DTYPE t2 = (f2 == i3) ? 0 : inLocalBuff[f2];

            if(f2 == i3 || (f1 < i2 && t1 <= t2)) {
                indicesTemp[i] = inLocalIndices[f1];

                temp[i] = t1;
                f1++;
            } else {
                indicesTemp[i] = inLocalIndices[f2];

                assert(f2 < i3);
                temp[i] = t2;
                f2++;
            }
            if(f1 == i2 && f2 == i3) {
                f1 = i3;
                i2 += 2*width;
                i3 += 2*width;
                if(i2 >= MaxSliceLen) i2 = MaxSliceLen;
                if(i3 >= MaxSliceLen) i3 = MaxSliceLen;
                f2 = i2;
            }
        }

        LoopCopy:
        for(int i = 0; i < MaxSliceLen; i++) {
            #pragma HLS pipeline II=1
            #pragma HLS unroll factor=CONFIG_M_AXI_WIDTH

            inLocalBuff[i] = temp[i];
            inLocalIndices[i] = indicesTemp[i];
        }
    }
}



/**
 * @brief      The sub-function to feed data from global memory into the PEs.
 *             This kernel supports burst read/write.
 *
 * @param[in]  inputTn        The input tn
 * @param      streamInputTn  The stream connected to the first PE
 * @param[in]  dim0           The dim 0
 * @param[in]  dim1           The dim 1
 * @param[in]  vecsPerSlice   The number of vectors(of length of m_axi_width) per input slice
 */
void UnitReadInput(
    const MemoryPackF_t *inputTn,
    Stream<MemoryPackF_t, PipeDepth> &streamInputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice){

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(dim1%CONFIG_M_AXI_WIDTH==0);

    unsigned indxS;
    const unsigned safeDim0 = dim0 - dim0 % UnitCount;
    const unsigned remDim0 = dim0 % UnitCount;

    // 1. Handle safe part of data and write out data in burst.
    For_Main: 
    for(unsigned batch=0; batch<safeDim0; batch+=UnitCount){
        LoopPEs:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE:
            for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
                #pragma HLS PIPELINE II=1

                const unsigned d0 = batch+iPE;
                indxS = (d0)*vecsPerSlice + iVec;
                streamInputTn.Push(inputTn[indxS]);
            }
        }
    }

    // 2. Handle remainder of data without burst writes.
    if(remDim0!=0){
        LoopPEs_Rem:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE_Rem:
            for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
                #pragma HLS PIPELINE II=1
                
                const unsigned d0 = safeDim0+iPE;
                MemoryPackF_t vec;
                if(d0<dim0){
                    indxS = (d0)*vecsPerSlice + iVec;
                    vec = inputTn[indxS];
                }else{
                    vec.Fill(0.0f);
                }
                streamInputTn.Push(vec);
            }
        }

    }

}

/**
 * @brief      The sub-function to handle the data produced by PEs.
 *             The output tensor should be padded in the last dimnesion such that shape[-1]%m_axi_width=0
 *             This kernel supports burst read/write.
 *
 * @param      indicesSplitedTn    The indices splited tn (batchsize x K)
 * @param      streamIndices       The stream connected to the first PE that outputs results produced by all the PEs.
 * @param[in]  dim0                The dim 0
 * @param[in]  dim1                The dim 1
 * @param[in]  vecsPerOutputSlice  The number of vectors(of length of m_axi_width) per output slice
 */
void UnitWriteOutput(
    MemoryPackI_t *indicesSplitedTn,
    Stream<MemoryPackI_t, PipeDepth> &streamIndices,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerOutputSlice){

    /*
    // FIFO depth should be greater or equal to 'vecsPerOutputSlice' for PE's not to stall.
    assert(PipeDepth>=vecsPerOutputSlice);
    unsigned indxD;
    For_Main: 
    for(unsigned batch=0; batch<dim0; batch+=UnitCount){
        LoopPEs:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE:
            for(unsigned iVec=0; iVec<vecsPerOutputSlice; iVec++){
                const unsigned d0 = batch+iPE;
                MemoryPackI_t vec = streamIndices.Pop();
                if(d0<dim0){
                    indxD = (d0)*vecsPerOutputSlice + iVec;
                    indicesSplitedTn[indxD] = vec;
                }
            }
        }
    }
    */


    // FIFO depth should be greater or equal to 'vecsPerOutputSlice' for PE's not to stall.
    assert(PipeDepth>=vecsPerOutputSlice);
    unsigned indxD;
    const unsigned safeDim0 = dim0 - dim0 % UnitCount;
    const unsigned remDim0 = dim0 % UnitCount;

    // 1. Handle safe part of data and write out data in burst.
    For_Main: 
    for(unsigned batch=0; batch<safeDim0; batch+=UnitCount){
        LoopPEs:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE:
            for(unsigned iVec=0; iVec<vecsPerOutputSlice; iVec++){
                #pragma HLS PIPELINE II=1

                const unsigned d0 = batch+iPE;
                indxD = (d0)*vecsPerOutputSlice + iVec;
                indicesSplitedTn[indxD] = streamIndices.Pop();
            }
        }
    }

    // 2. Handle remainder of data without burst writes.
    if(remDim0!=0){
        LoopPEs_Rem:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE_Rem:
            for(unsigned iVec=0; iVec<vecsPerOutputSlice; iVec++){
                const unsigned d0 = safeDim0+iPE;
                MemoryPackI_t vec = streamIndices.Pop();
                if(d0<dim0){
                    indxD = (d0)*vecsPerOutputSlice + iVec;
                    indicesSplitedTn[indxD] = vec;
                }
            }
        }
    }

}

/**
 * @brief      The sub-function for PEs.
 *             Each PE handles a single input slice that is streamed into its local buffer through other PEs.
 *
 * @param      streamDataIn        The data input stream that is connected to UnitReadInput or the previous PE's data output stream.
 * @param      streamDataOut       The data output stream that is connected to nowhere or the next PE's data input stream.
 * @param      streamIndicesIn     The indices input stream that is connected to nowhere or the next PE's indices output stream.
 * @param      streamIndicesOut    The indices output stream that is connected to UnitWriteOutput or the previous PE's indices input stream.
 * @param[in]  dim0                The dim 0
 * @param[in]  dim1                The dim 1
 * @param[in]  vecsPerSlice        The number of vectors(of length of m_axi_width) per input slice
 * @param[in]  vecsPerOutputSlice  The number of vectors(of length of m_axi_width) per output slice
 * @param[in]  kValue              The k value
 * @param[in]  unitIndex           The unit index
 */
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
    assert(dim1==MaxSliceLen);

    // FIFO depth should be greater or equal to 'vecsPerOutputSlice' for PE's not to stall.
    assert(PipeDepth>=vecsPerOutputSlice);

    unsigned min_idx;
    unsigned indxS,indxD;

    CONFIG_DTYPE sliceData[MaxSliceLen];
#pragma HLS RESOURCE variable=sliceData core=RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable=sliceData cyclic factor=CONFIG_M_AXI_WIDTH dim=1

    ///TODO: Change the type to a 10-bit ap_uint to reduce resource usage.
    unsigned sliceIndices[MaxSliceLen];
//#pragma HLS RESOURCE variable=sliceIndices core=RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable=sliceIndices cyclic factor=CONFIG_M_AXI_WIDTH dim=1

    //MemoryPackF_t sliceVec;
    MemoryPackI_t outputCache;
    unsigned outputCacheVecSubIdx;

    constexpr unsigned tripCountLoopMain = 5 * 1024 / UnitCount;

    LoopMain: for(unsigned batch=0; batch<dim0; batch+=UnitCount){
#pragma HLS LOOP_TRIPCOUNT min=tripCountLoopMain max=tripCountLoopMain
        //--------------------------------------------------
        // 1. Read the current slice and indices into the local memory.
        

        /*
        // This method of data handling is not recommended as it forces 
        // the data transfers in UnitRead to not be bursts.

        LoopReadSlice: 
        for(unsigned idx=0; idx<vecsPerSlice; idx++){
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64
            
            LoopInputPass01:
            for(unsigned iPE=0; iPE<UnitCount-unitIndex; iPE++){
                #pragma HLS PIPELINE II=1
                MemoryPackF_t vec = streamDataIn.Pop();
                if(iPE>0){
                    if(unitIndex<(UnitCount-1)){
                        streamDataOut.Push(vec);
                    }
                }else{
                    sliceVec = vec;
                }
            }
            
            const unsigned offsetLocal = idx*CONFIG_M_AXI_WIDTH;

            LoopReadUnroll1:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                const unsigned indxLocal = offsetLocal+i; 
                sliceData[indxLocal] = sliceVec[i];
                sliceIndices[indxLocal] = indxLocal;
            }
        }
        */

        LoopInputPass01:
        for(unsigned iPE=0; iPE<UnitCount-unitIndex; iPE++){

            // It's better to read the whole slice at once, this way memory accesses 
            // in UnitRead will be burst transfers.
            LoopReadSlice: 
            for(unsigned idx=0; idx<vecsPerSlice; idx++){
                #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                #pragma HLS PIPELINE II=1

                MemoryPackF_t vec = streamDataIn.Pop();
                if(iPE>0){
                    if(unitIndex<(UnitCount-1)){
                        streamDataOut.Push(vec);
                    }
                }else{
                    const unsigned offsetLocal = idx*CONFIG_M_AXI_WIDTH;
                    LoopReadUnroll1:
                    for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                        #pragma HLS UNROLL
                        const unsigned indxLocal = offsetLocal+i; 
                        sliceData[indxLocal] = vec[i];
                        sliceIndices[indxLocal] = indxLocal;
                    }
                }
            }
            
        }

        //--------------------------------------------------
        // 2. Run sorting algorithm on the local memory.
        MergeSortWithIndices(sliceData, sliceIndices);

        LoopPushTheResults:
        for(unsigned i=0; i<kValue; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=20 max=20
            outputCacheVecSubIdx = i % CONFIG_M_AXI_WIDTH;
            outputCache[outputCacheVecSubIdx] = sliceIndices[i];
            if (outputCacheVecSubIdx == (CONFIG_M_AXI_WIDTH - 1) || i == (kValue - 1)) {
                streamIndicesOut.Push(outputCache);
#ifdef KERNEL_LOGS
                cout << "PE" << unitIndex << ": " << " Sorted Vec i=" << i << endl;
#endif
            }
        }
        
        //--------------------------------------------------
        
        // 3. Handle incoming data of streamIndicesIn from the other PEs.
        const unsigned _len2 = UnitCount-unitIndex-1;
        LoopHandleOtherPEsOutput:
        for(unsigned iPE=0; iPE<_len2; iPE++){
            #pragma HLS LOOP_TRIPCOUNT min=UnitCount max=UnitCount
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

/**
 * @brief      Finds the least k elements for every slice in batch.
 *             The top-function of the kernel.
 *             Supports handling an input tensor with "dim0 % UnitCount != 0".
 *             The latency will be reported for [5x1024]x1024, k=20, PEs=UnitCount, m_axi_width=16, pipe_depth=2
 *             This kernel supports burst read/write.
 *
 * @param[in]  inputTn             The input tn
 * @param[out] indicesSplitedTn    The indices splited tn (the tensor with a buffer of length of batchsize x kValue)
 * @param[in]  dim0                The dim 0
 * @param[in]  dim1                The dim 1
 * @param[in]  kValue              The k value
 * @param[in]  vecsPerSlice        The number of vectors(of length of m_axi_width) per input slice
 * @param[in]  vecsPerOutputSlice  The number of vectors(of length of m_axi_width) per output slice
 */
void task_topk(
        const MemoryPackF_t *inputTn,
        MemoryPackI_t *indicesSplitedTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned kValue,
        const unsigned vecsPerSlice,
        const unsigned vecsPerOutputSlice){

#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=2
#pragma HLS INTERFACE m_axi port=indicesSplitedTn offset=slave bundle=gmem2 max_read_burst_length=2 max_write_burst_length=16
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

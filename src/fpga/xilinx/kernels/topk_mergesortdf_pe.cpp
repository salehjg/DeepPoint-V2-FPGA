 #include <cassert>
#include <iostream>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "AxiHelper.h"
#include "xilinx/config.h"
#include "ap_int.h"

using namespace std;
using namespace ConfigTaskTopK;
using hlslib::Stream;

constexpr unsigned latencyReportBatchSize = 5 * 1024;

struct PairDataIndex_t{
public:
    PairDataIndex_t(){
        // This constructor should NOT initialize data and index
        // as it causes the vivado hls dataflow form checks to fail.
    }

    PairDataIndex_t(CONFIG_DTYPE data, unsigned index){
        this->data = data;
        this->index = index;
    }
    CONFIG_DTYPE data;
    ap_uint<ConstexperCeilLog2(MaxSliceLen)> index;
    //unsigned index;
};

void TopK_MergeSortDF_V1_UnitRead(
    const MemoryPackF_t *inputTn,
    Stream<PairDataIndex_t, 8> streamDataOut[UnitCount][2], //[U][0]: Left, [U][1]: Right
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice){

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(dim1==MaxSliceLen);
    constexpr unsigned tripCountLoopBatch = latencyReportBatchSize / UnitCount;

    const unsigned safeDim0 = dim0 - dim0 % UnitCount;
    const unsigned remDim0 = dim0 % UnitCount;

    // Safe (BURST IO)
    LoopBatch:
    for(unsigned batch=0; batch<safeDim0; batch+=UnitCount){
        #pragma HLS LOOP_TRIPCOUNT min=tripCountLoopBatch max=tripCountLoopBatch
        LoopPEs:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE:
            for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
                #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                #pragma HLS PIPELINE II=1

                const unsigned d0 = batch+iPE;
                const unsigned indxS = d0*vecsPerSlice + iVec;
                MemoryPackF_t vec = inputTn[indxS];

                LoopPush:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    PairDataIndex_t pair(vec[i], iVec*CONFIG_M_AXI_WIDTH+i);
                    if(i%2==0){
                        streamDataOut[iPE][0].Push(pair);
                    }else{
                        streamDataOut[iPE][1].Push(pair);
                    }
                }
            }
        }
    }

    // 2. Handle remainder of data without burst writes.
    if(remDim0!=0){

        LoopPEs_Rem:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE_Rem:
            for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
                #pragma HLS LOOP_TRIPCOUNT min=64 max=64
                #pragma HLS PIPELINE II=1

                const unsigned d0 = safeDim0+iPE;
                
                MemoryPackF_t vec;
                if(d0<dim0){
                    const unsigned indxS = d0*vecsPerSlice + iVec;
                    vec = inputTn[indxS];
                }else{
                    vec.Fill(0.0f);
                }

                LoopPush_Rem:
                for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    PairDataIndex_t pair(vec[i], iVec*CONFIG_M_AXI_WIDTH+i);
                    if(i%2==0){
                        streamDataOut[iPE][0].Push(pair);
                    }else{
                        streamDataOut[iPE][1].Push(pair);
                    }
                }
            }
        }
        
    }
}

void TopK_MergeSortDF_V1_UnitWrite(
    MemoryPackI_t *indicesSplitedTn,
    Stream<PairDataIndex_t, MaxK> streamDataInL[UnitCount],
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerOutputSlice){

    assert(dim1==MaxSliceLen);
    const unsigned safeDim0 = dim0 - dim0 % UnitCount;
    const unsigned remDim0 = dim0 % UnitCount;
    constexpr unsigned tripCountLoopBatch = latencyReportBatchSize / UnitCount;

    // 1. Handle safe part of data and write out data in burst.
    LoopBatch:
    for(unsigned batch=0; batch<safeDim0; batch+=UnitCount) {
        #pragma HLS LOOP_TRIPCOUNT min=tripCountLoopBatch max=tripCountLoopBatch
        LoopPEs:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE:
            for (unsigned iVec = 0; iVec < vecsPerOutputSlice; iVec++) {
                #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                #pragma HLS PIPELINE II=1

                const unsigned d0 = batch+iPE;
                const unsigned indxD = d0*vecsPerOutputSlice + iVec;
                MemoryPackI_t vec;

                LoopPush:
                for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                    #pragma HLS UNROLL
                    PairDataIndex_t result = streamDataInL[iPE].Pop();
                    vec[i] = result.index;
                }

                indicesSplitedTn[indxD] = vec;
            }
        }
    }

    // 2. Handle remainder of data without burst writes.
    if(remDim0!=0){
        LoopPEs_Rem:
        for(unsigned iPE=0; iPE<UnitCount; iPE++){
            LoopVecsPerPE_Rem:
            for (unsigned iVec = 0; iVec < vecsPerOutputSlice; iVec++) {
                #pragma HLS LOOP_TRIPCOUNT min=2 max=2
                #pragma HLS PIPELINE II=1

                const unsigned d0 = safeDim0+iPE;
                MemoryPackI_t vec;

                LoopPush_Rem:
                for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                    #pragma HLS UNROLL
                    PairDataIndex_t result = streamDataInL[iPE].Pop();
                    vec[i] = result.index;
                }
                if(d0<dim0){
                    const unsigned indxD = d0*vecsPerOutputSlice + iVec;
                    indicesSplitedTn[indxD] = vec;
                }
            }
        }
    }
}


template<
    unsigned windowWidth, // 2,4,8,...,512
    unsigned depthInputs,
    unsigned depthOutputs>
void TopK_MergeSortDF_V1_UnitMergeX(
    Stream<PairDataIndex_t, depthInputs> &streamDataInL,
    Stream<PairDataIndex_t, depthInputs> &streamDataInR,
    Stream<PairDataIndex_t, depthOutputs> &streamDataOutL,
    Stream<PairDataIndex_t, depthOutputs> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    assert(dim1 % windowWidth == 0);
    assert(dim1 <= MaxSliceLen);

    constexpr unsigned win2 = (2 * windowWidth);
    const unsigned pairsToBeMerged = MaxSliceLen / win2;

    constexpr unsigned tripCountLoopBatch = latencyReportBatchSize / UnitCount;

    LoopBatch:
    for(unsigned batch=0; batch<dim0; batch+=UnitCount) {
        #pragma HLS LOOP_TRIPCOUNT min=tripCountLoopBatch max=tripCountLoopBatch
        // It doesn't matter that 'batch' starts from zero instead of iPE.

        LoopPairs:
        for (unsigned pair = 0; pair < pairsToBeMerged; pair++) {
            PairDataIndex_t lastFetch1, lastFetch2;

            // ---------------------------------------------
            // 1. Merge the pairs
            int f1 = 0, lastF1 = -1;
            int f2 = windowWidth, lastF2 = -1;
            unsigned i2 = windowWidth;
            unsigned i3 = win2;
            if (i2 >= win2) i2 = win2;
            if (i3 >= win2) i3 = win2;

            LoopMerge1:
            for (unsigned i = 0; i < win2; i++) {
                #pragma HLS PIPELINE II=1

                PairDataIndex_t t1, t2;

                if(lastF1 != f1 && f1<i2){
                    lastFetch1 = streamDataInL.Pop();
                    t1 = lastFetch1;
                    lastF1 = f1;
                }else{
                    t1 = lastFetch1;
                }

                if(lastF2 != f2 && f2<i3){
                    lastFetch2 = streamDataInR.Pop();
                    t2 = lastFetch2;
                    lastF2 = f2;
                }else{
                    t2 = lastFetch2;
                }

                if (f2 == i3 || (f1 < i2 && t1.data <= t2.data)){
                    if (pair % 2 == 0) {
                        streamDataOutL.Push(t1);
                    } else {
                        streamDataOutR.Push(t1);
                    }
                    f1++;
                }else{
                    assert(f2 < i3);
                    if (pair % 2 == 0) {
                        streamDataOutL.Push(t2);
                    } else {
                        streamDataOutR.Push(t2);
                    }
                    f2++;
                }
            }
        }
    }
}

template<
    unsigned windowWidth, // 512 only
    unsigned depthInputs,
    unsigned depthOutput>
void TopK_MergeSortDF_V1_UnitMergeLast(
    Stream<PairDataIndex_t, depthInputs> &streamDataInL,
    Stream<PairDataIndex_t, depthInputs> &streamDataInR,
    Stream<PairDataIndex_t, depthOutput> &streamDataOutL,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerOutputSlice){

    assert(dim1 % windowWidth == 0);
    assert(dim1 <= MaxSliceLen);

    constexpr unsigned win2 = (2 * windowWidth);
    const unsigned pairsToBeMerged = MaxSliceLen / win2;
    const unsigned kValuePadded = vecsPerOutputSlice * CONFIG_M_AXI_WIDTH;

    constexpr unsigned tripCountLoopBatch = 5 * 1024 / UnitCount;

    LoopBatch:
    for(unsigned batch=0; batch<dim0; batch+=UnitCount) {
        #pragma HLS LOOP_TRIPCOUNT min=tripCountLoopBatch max=tripCountLoopBatch
        // It doesn't matter that 'batch' starts from zero instead of iPE.

        LoopPairs:
        for (unsigned pair = 0; pair < pairsToBeMerged; pair++) {
            PairDataIndex_t lastFetch1, lastFetch2;

            // ---------------------------------------------
            // 1. Merge the pairs
            int f1 = 0, lastF1 = 1;
            int f2 = windowWidth, lastF2 = windowWidth+1;
            unsigned i2 = windowWidth;
            unsigned i3 = win2;
            if (i2 >= win2) i2 = win2;
            if (i3 >= win2) i3 = win2;

            LoopMerge1:
            for (unsigned i = 0; i < win2; i++) {
            #pragma HLS PIPELINE II=1

                PairDataIndex_t t1, t2;

                if(lastF1 != f1 && f1<i2){
                    lastFetch1 = streamDataInL.Pop();
                    t1 = lastFetch1;
                    lastF1 = f1;
                }else{
                    t1 = lastFetch1;
                }

                if(lastF2 != f2 && f2<i3){
                    lastFetch2 = streamDataInR.Pop();
                    t2 = lastFetch2;
                    lastF2 = f2;
                }else{
                    t2 = lastFetch2;
                }

                if (f2 == i3 || (f1 < i2 && t1.data <= t2.data)) {
                    if (i < kValuePadded) {
                        streamDataOutL.Push(t1);
                    }
                    f1++;
                }else{
                    assert(f2 < i3);
                    if (i < kValuePadded) {
                        streamDataOutL.Push(t2);
                    }
                    f2++;
                }
            }
        }
    }
}



void TopK_MergeSortDF_V1_UnitMerge1(
    Stream<PairDataIndex_t, 8> &streamDataInL,
    Stream<PairDataIndex_t, 8> &streamDataInR,
    Stream<PairDataIndex_t, 2> &streamDataOutL,
    Stream<PairDataIndex_t, 2> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 1;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, 8, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge2(
    Stream<PairDataIndex_t, 2> &streamDataInL,
    Stream<PairDataIndex_t, 2> &streamDataInR,
    Stream<PairDataIndex_t, 4> &streamDataOutL,
    Stream<PairDataIndex_t, 4> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 2;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge4(
    Stream<PairDataIndex_t, 4> &streamDataInL,
    Stream<PairDataIndex_t, 4> &streamDataInR,
    Stream<PairDataIndex_t, 8> &streamDataOutL,
    Stream<PairDataIndex_t, 8> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 4;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge8(
    Stream<PairDataIndex_t, 8> &streamDataInL,
    Stream<PairDataIndex_t, 8> &streamDataInR,
    Stream<PairDataIndex_t, 16> &streamDataOutL,
    Stream<PairDataIndex_t, 16> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 8;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge16(
    Stream<PairDataIndex_t, 16> &streamDataInL,
    Stream<PairDataIndex_t, 16> &streamDataInR,
    Stream<PairDataIndex_t, 32> &streamDataOutL,
    Stream<PairDataIndex_t, 32> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 16;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge32(
    Stream<PairDataIndex_t, 32> &streamDataInL,
    Stream<PairDataIndex_t, 32> &streamDataInR,
    Stream<PairDataIndex_t, 64> &streamDataOutL,
    Stream<PairDataIndex_t, 64> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 32;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge64(
    Stream<PairDataIndex_t, 64> &streamDataInL,
    Stream<PairDataIndex_t, 64> &streamDataInR,
    Stream<PairDataIndex_t, 128> &streamDataOutL,
    Stream<PairDataIndex_t, 128> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 64;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge128(
    Stream<PairDataIndex_t, 128> &streamDataInL,
    Stream<PairDataIndex_t, 128> &streamDataInR,
    Stream<PairDataIndex_t, 256> &streamDataOutL,
    Stream<PairDataIndex_t, 256> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 128;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge256(
    Stream<PairDataIndex_t, 256> &streamDataInL,
    Stream<PairDataIndex_t, 256> &streamDataInR,
    Stream<PairDataIndex_t, 512> &streamDataOutL,
    Stream<PairDataIndex_t, 512> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1){

    constexpr unsigned windowWidth = 256;

    TopK_MergeSortDF_V1_UnitMergeX<windowWidth, windowWidth*1, windowWidth*2>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        streamDataOutR,
        dim0,
        dim1);
}

void TopK_MergeSortDF_V1_UnitMerge512(
    Stream<PairDataIndex_t, 512> &streamDataInL,
    Stream<PairDataIndex_t, 512> &streamDataInR,
    Stream<PairDataIndex_t, MaxK> &streamDataOutL,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerOutputSlice){

    constexpr unsigned windowWidth = 512;

    TopK_MergeSortDF_V1_UnitMergeLast<windowWidth, windowWidth*1, MaxK>(
        streamDataInL,
        streamDataInR,
        streamDataOutL,
        dim0,
        dim1,
        vecsPerOutputSlice);
}

void TopK_MergeSortDF_V1(
        const MemoryPackF_t *inputTn,
        MemoryPackI_t *indicesSplitedTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned kValue,
        const unsigned vecsPerSlice,
        const unsigned vecsPerOutputSlice){
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
    cout<<"Number of PEs: "<< UnitCount<<endl;
#endif

    #pragma HLS DATAFLOW

    Stream<PairDataIndex_t, 8> streamRead_W1[UnitCount][2];
//#pragma HLS RESOURCE variable=streamRead_W1 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamRead_W1 depth=8

    Stream<PairDataIndex_t, 2> streamW1_W2[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW1_W2 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=2

    Stream<PairDataIndex_t, 4> streamW2_W4[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW2_W4 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=4

    Stream<PairDataIndex_t, 8> streamW4_W8[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW4_W8 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=8

    Stream<PairDataIndex_t, 16> streamW8_W16[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW8_W16 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=16

    Stream<PairDataIndex_t, 32> streamW16_W32[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW16_W32 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=32

    Stream<PairDataIndex_t, 64> streamW32_W64[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW32_W64 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=64

    Stream<PairDataIndex_t, 128> streamW64_W128[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW64_W128 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=128

    Stream<PairDataIndex_t, 256> streamW128_W256[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW128_W256 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=256

    Stream<PairDataIndex_t, 512> streamW256_W512[UnitCount][2];
//#pragma HLS RESOURCE variable=streamW256_W512 core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=512

    Stream<PairDataIndex_t, 20> streamW512_Write[UnitCount];
//#pragma HLS RESOURCE variable=streamW512_Write core=FIFO_LUTRAM
#pragma HLS STREAM variable=streamReadToW1 depth=20


#ifndef HLSLIB_SYNTHESIS
    // Name the arrays of channels for debugging purposes
    for(unsigned iPE=0; iPE<UnitCount; iPE++){
        for(unsigned i=0; i<2; i++) {
            streamRead_W1[iPE][i].set_name(("streamRead_W1["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW1_W2[iPE][i].set_name(("streamW1_W2["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW2_W4[iPE][i].set_name(("streamW2_W4["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW4_W8[iPE][i].set_name(("streamW4_W8["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW8_W16[iPE][i].set_name(("streamW8_W16["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW16_W32[iPE][i].set_name(("streamW16_W32["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW32_W64[iPE][i].set_name(("streamW32_W64["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW64_W128[iPE][i].set_name(("streamW64_W128["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW128_W256[iPE][i].set_name(("streamW128_W256["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
            streamW256_W512[iPE][i].set_name(("streamW256_W512["+ std::to_string(iPE)+"]["+std::to_string(i)+"]").c_str());
        }
        streamW512_Write[iPE].set_name(("streamW512_WK["+ std::to_string(iPE)+"]").c_str());
    }
#endif

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitRead, 
        inputTn, 
        streamRead_W1, 
        dim0, 
        dim1, 
        vecsPerSlice);

    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitWrite,
        indicesSplitedTn,
        streamW512_Write,
        dim0, 
        dim1,
        vecsPerOutputSlice);

    for(unsigned iPE=0; iPE<UnitCount; iPE++){
        #pragma HLS UNROLL

        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge1,
            streamRead_W1[iPE][0],
            streamRead_W1[iPE][1],
            streamW1_W2[iPE][0],
            streamW1_W2[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge2,
            streamW1_W2[iPE][0],
            streamW1_W2[iPE][1],
            streamW2_W4[iPE][0],
            streamW2_W4[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge4,
            streamW2_W4[iPE][0],
            streamW2_W4[iPE][1],
            streamW4_W8[iPE][0],
            streamW4_W8[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge8,
            streamW4_W8[iPE][0],
            streamW4_W8[iPE][1],
            streamW8_W16[iPE][0],
            streamW8_W16[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge16,
            streamW8_W16[iPE][0],
            streamW8_W16[iPE][1],
            streamW16_W32[iPE][0],
            streamW16_W32[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge32,
            streamW16_W32[iPE][0],
            streamW16_W32[iPE][1],
            streamW32_W64[iPE][0],
            streamW32_W64[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge64,
            streamW32_W64[iPE][0],
            streamW32_W64[iPE][1],
            streamW64_W128[iPE][0],
            streamW64_W128[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge128,
            streamW64_W128[iPE][0],
            streamW64_W128[iPE][1],
            streamW128_W256[iPE][0],
            streamW128_W256[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge256,
            streamW128_W256[iPE][0],
            streamW128_W256[iPE][1],
            streamW256_W512[iPE][0],
            streamW256_W512[iPE][1],
            dim0,
            dim1);
        
        HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge512,
            streamW256_W512[iPE][0],
            streamW256_W512[iPE][1],
            streamW512_Write[iPE],
            dim0,
            dim1,
            vecsPerOutputSlice);

    }
    HLSLIB_DATAFLOW_FINALIZE();

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

    static_assert(MaxSliceLen==1024, "Only MaxSliceLen=1024 is supported.");
    
    TopK_MergeSortDF_V1(inputTn, indicesSplitedTn, dim0, dim1, kValue, vecsPerSlice, vecsPerOutputSlice);
}
}

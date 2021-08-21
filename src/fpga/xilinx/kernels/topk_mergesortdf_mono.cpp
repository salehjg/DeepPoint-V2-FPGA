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
    Stream<PairDataIndex_t, 8> &streamDataOutL,
    Stream<PairDataIndex_t, 8> &streamDataOutR,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerSlice){

    assert(dim1==MaxSliceLen);

    LoopBatch:
    for(unsigned batch=0; batch<dim0; batch++){
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        LoopVecsPerPE:
        for(unsigned iVec=0; iVec<vecsPerSlice; iVec++){
            #pragma HLS LOOP_TRIPCOUNT min=64 max=64
            #pragma HLS PIPELINE II=1

            const unsigned indxS = batch*vecsPerSlice + iVec;
            MemoryPackF_t vec = inputTn[indxS];

            LoopPush:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                PairDataIndex_t pair(vec[i], iVec*CONFIG_M_AXI_WIDTH+i);
                if(i%2==0){
                    streamDataOutL.Push(pair);
                }else{
                    streamDataOutR.Push(pair);
                }
            }
        }
    }
}

void TopK_MergeSortDF_V1_UnitWrite(
    MemoryPackI_t *indicesSplitedTn,
    Stream<PairDataIndex_t, 1024> &streamDataInL,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerOutputSlice){

    assert(dim1==MaxSliceLen);

    LoopBatch:
    for(unsigned batch=0; batch<dim0; batch++) {
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        LoopVecsPerPE:
        for (unsigned iVec = 0; iVec < vecsPerOutputSlice; iVec++) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
#pragma HLS PIPELINE II=1

            const unsigned indxD = batch*vecsPerOutputSlice + iVec;
            MemoryPackI_t vec;

            LoopPush:
            for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
#pragma HLS UNROLL
                PairDataIndex_t result = streamDataInL.Pop();
                vec[i] = result.index;
            }

            indicesSplitedTn[indxD] = vec;
        }
    }
}

/*
template<
     unsigned windowWidth, // 2,4,8,...,512
     unsigned depthInputs,
     unsigned depthOutputs
>
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

    LoopBatch:
    for(unsigned batch=0; batch<dim0; batch++) {
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        LoopPairs:
        for (unsigned pair = 0; pair < pairsToBeMerged; pair++) {
            PairDataIndex_t buffPair[win2];
#pragma HLS ARRAY_PARTITION variable=buffPair complete dim=1

            // ---------------------------------------------
            // 1. Fetch two pairs before merging
            LoopFetchPairs:
            for (unsigned w = 0; w < win2; w++) {
                if (w < windowWidth) {
                    buffPair[w] = streamDataInL.Pop();
                } else {
                    buffPair[w] = streamDataInR.Pop();
                }
            }

            // ---------------------------------------------
            // 2. Merge the pairs
            unsigned f1 = 0;
            unsigned f2 = windowWidth;
            unsigned i2 = windowWidth;
            unsigned i3 = win2;
            if (i2 >= win2) i2 = win2;
            if (i3 >= win2) i3 = win2;

            LoopMerge1:
            for (unsigned i = 0; i < win2; i++) {
#pragma HLS PIPELINE II=1

                PairDataIndex_t t1 = buffPair[f1];
                //PairDataIndex_t t2 = (f2 == i3) ? PairDataIndex_t(0, buffPair[f2].index) : buffPair[f2];
                PairDataIndex_t t2 = buffPair[f2];


                if (f2 == i3 || (f1 < i2 && t1.data <= t2.data)) {
                    if (pair % 2 == 0) {
                        streamDataOutL.Push(t1);
                    } else {
                        streamDataOutR.Push(t1);
                    }
                    f1++;
                } else {
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
    unsigned depthOutput
>
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

    LoopBatch:
    for(unsigned batch=0; batch<dim0; batch++) {
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
        LoopPairs:
        for (unsigned pair = 0; pair < pairsToBeMerged; pair++) {
            PairDataIndex_t buffPair[win2];

            // ---------------------------------------------
            // 1. Fetch two pairs before merging
            LoopFetchPairs:
            for (unsigned w = 0; w < win2; w++) {
                if (w < windowWidth) {
                    buffPair[w] = streamDataInL.Pop();
                } else {
                    buffPair[w] = streamDataInR.Pop();
                }
            }

            // ---------------------------------------------
            // 2. Merge the pairs
            unsigned f1 = 0;
            unsigned f2 = windowWidth;
            unsigned i2 = windowWidth;
            unsigned i3 = win2;
            if (i2 >= win2) i2 = win2;
            if (i3 >= win2) i3 = win2;

            LoopMerge1:
            for (unsigned i = 0; i < win2; i++) {
#pragma HLS PIPELINE II=1

                PairDataIndex_t t1 = buffPair[f1];
                //PairDataIndex_t t2 = (f2 == i3) ? PairDataIndex_t(0, buffPair[f2].index) : buffPair[f2];
                PairDataIndex_t t2 = buffPair[f2];

                if (f2 == i3 || (f1 < i2 && t1.data <= t2.data)) {
                    if (i < kValuePadded) {
                        streamDataOutL.Push(t1);
                    }
                    f1++;
                } else {
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
*/




 template<
         unsigned windowWidth, // 2,4,8,...,512
         unsigned depthInputs,
         unsigned depthOutputs
 >
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

     LoopBatch:
     for(unsigned batch=0; batch<dim0; batch++) {
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
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

                 if (f2 == i3 || (f1 < i2 && t1.data <= t2.data)) {
                     if (pair % 2 == 0) {
                         streamDataOutL.Push(t1);
                     } else {
                         streamDataOutR.Push(t1);
                     }
                     f1++;
                 } else {
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
         unsigned depthOutput
 >
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

     LoopBatch:
     for(unsigned batch=0; batch<dim0; batch++) {
#pragma HLS LOOP_TRIPCOUNT min=5120 max=5120
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
                 } else {
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
    Stream<PairDataIndex_t, 1024> &streamDataOutL,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned vecsPerOutputSlice){

    constexpr unsigned windowWidth = 512;

    TopK_MergeSortDF_V1_UnitMergeLast<windowWidth, windowWidth*1, windowWidth*2>(
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
        //MemoryPackI_t *indicesSplitedTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned kValue,
        const unsigned vecsPerSlice,
        const unsigned vecsPerOutputSlice){
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl; 
#endif

    #pragma HLS DATAFLOW

    Stream<PairDataIndex_t, 8> streamRead_W1[2];
#pragma HLS STREAM variable=streamRead_W1 depth=8

    Stream<PairDataIndex_t, 2> streamW1_W2[2];
#pragma HLS STREAM variable=streamReadToW1 depth=2

    Stream<PairDataIndex_t, 4> streamW2_W4[2];
#pragma HLS STREAM variable=streamReadToW1 depth=4

    Stream<PairDataIndex_t, 8> streamW4_W8[2];
#pragma HLS STREAM variable=streamReadToW1 depth=8

    Stream<PairDataIndex_t, 16> streamW8_W16[2];
#pragma HLS STREAM variable=streamReadToW1 depth=16

    Stream<PairDataIndex_t, 32> streamW16_W32[2];
#pragma HLS STREAM variable=streamReadToW1 depth=32

    Stream<PairDataIndex_t, 64> streamW32_W64[2];
#pragma HLS STREAM variable=streamReadToW1 depth=64

    Stream<PairDataIndex_t, 128> streamW64_W128[2];
#pragma HLS STREAM variable=streamReadToW1 depth=128

    Stream<PairDataIndex_t, 256> streamW128_W256[2];
#pragma HLS STREAM variable=streamReadToW1 depth=256

    Stream<PairDataIndex_t, 512> streamW256_W512[2];
#pragma HLS STREAM variable=streamReadToW1 depth=512

    Stream<PairDataIndex_t, 1024> streamW512_Write;
#pragma HLS STREAM variable=streamReadToW1 depth=1024


#ifndef HLSLIB_SYNTHESIS
    // Name the arrays of channels for debugging purposes
    for (unsigned i = 0; i < 2; i++) {
        streamRead_W1[i].set_name(("streamRead_W1[" + std::to_string(i) + "]").c_str());
        streamW1_W2[i].set_name(("streamW1_W2[" + std::to_string(i) + "]").c_str());
        streamW2_W4[i].set_name(("streamW2_W4[" + std::to_string(i) + "]").c_str());
        streamW4_W8[i].set_name(("streamW4_W8[" + std::to_string(i) + "]").c_str());
        streamW8_W16[i].set_name(("streamW8_W16[" + std::to_string(i) + "]").c_str());
        streamW16_W32[i].set_name(("streamW16_W32[" + std::to_string(i) + "]").c_str());
        streamW32_W64[i].set_name(("streamW32_W64[" + std::to_string(i) + "]").c_str());
        streamW64_W128[i].set_name(("streamW64_W128[" + std::to_string(i) + "]").c_str());
        streamW128_W256[i].set_name(("streamW128_W256[" + std::to_string(i) + "]").c_str());
        streamW256_W512[i].set_name(("streamW256_W512[" + std::to_string(i) + "]").c_str());
    }
    streamW512_Write.set_name("streamW512_Write");
#endif

    HLSLIB_DATAFLOW_INIT();

    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitRead, 
        inputTn, 
        streamRead_W1[0], 
        streamRead_W1[1], 
        dim0, 
        dim1, 
        vecsPerSlice);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge1,
        streamRead_W1[0],
        streamRead_W1[1],
        streamW1_W2[0],
        streamW1_W2[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge2,
        streamW1_W2[0],
        streamW1_W2[1],
        streamW2_W4[0],
        streamW2_W4[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge4,
        streamW2_W4[0],
        streamW2_W4[1],
        streamW4_W8[0],
        streamW4_W8[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge8,
        streamW4_W8[0],
        streamW4_W8[1],
        streamW8_W16[0],
        streamW8_W16[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge16,
        streamW8_W16[0],
        streamW8_W16[1],
        streamW16_W32[0],
        streamW16_W32[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge32,
        streamW16_W32[0],
        streamW16_W32[1],
        streamW32_W64[0],
        streamW32_W64[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge64,
        streamW32_W64[0],
        streamW32_W64[1],
        streamW64_W128[0],
        streamW64_W128[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge128,
        streamW64_W128[0],
        streamW64_W128[1],
        streamW128_W256[0],
        streamW128_W256[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge256,
        streamW128_W256[0],
        streamW128_W256[1],
        streamW256_W512[0],
        streamW256_W512[1],
        dim0,
        dim1);
    
    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitMerge512,
        streamW256_W512[0],
        streamW256_W512[1],
        streamW512_Write,
        dim0,
        dim1,
        vecsPerOutputSlice);

    HLSLIB_DATAFLOW_FUNCTION(TopK_MergeSortDF_V1_UnitWrite,
        indicesSplitedTn,
        streamW512_Write, 
        dim0, 
        dim1,
        vecsPerOutputSlice);

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

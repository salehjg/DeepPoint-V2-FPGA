#pragma once

template<typename DType, int ReductionLen>
DType ParallelReduction1D(
    const DType *inputBuffer,
    const int len){

    DType lastResult=0;
    DType buff[ReductionLen];
#pragma HLS ARRAY_PARTITION variable=buff complete dim=0

    unsigned long iterations = ((len-1) / ReductionLen) + 1;
    unsigned long indxS, indxD;

    int tripLoopIter=1024/ReductionLen;

    LoopIter: for(unsigned long gIter=0; gIter<iterations; gIter++){
#pragma HLS LOOP_TRIPCOUNT min=tripLoopIter max=tripLoopIter
        // 1. read data into buff[:]
        LoopRead: for(int i=0;i<ReductionLen;i++){
#pragma HLS PIPELINE II=1
            indxS = gIter * ReductionLen + i;
            if(indxS<len){
                buff[i] = inputBuffer[indxS];
            }else{
                buff[i] = 0;
            }
            
        }

        //---------------------------------------
        LoopStride: for(int stride=ReductionLen/2; stride>0; stride/=2){
#pragma HLS PIPELINE II=1
            LoopReduce: for(int i=0; i<ReductionLen/2; i++){
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min=ReductionLen/2 max=ReductionLen/2
                if(i<stride){
                    buff[i] = buff[i] + buff[i+stride];
                }
            }
        }

        lastResult += buff[0];
    }

    return lastResult;
}

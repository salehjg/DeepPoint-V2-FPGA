#include <cassert>
#include <iostream>
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Stream.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std;
using hlslib::Stream;
using namespace ConfigTaskDataMover;

template<int bankIndex>
void DataMoverV2_UnitBankX(
        MemoryPackF_t *tensor,
        Stream<MemoryPackF_t, PipeDepth> &streamIn,
        Stream<MemoryPackF_t, PipeDepth> &streamOut,
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){

#pragma HLS INLINE

    // The conditional should be outside the pipelined loop, otherwise there wont be any burst accesses.
    if(srcBank == bankIndex){
        #ifdef KERNEL_LOGS
            cout<<"DataMoverV2_UnitBank"<<bankIndex<<": Reading "<<vecCount<<" vectors..."<<endl;
        #endif

        LoopRead:
        for(unsigned iter=0; iter<vecCount; iter++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            #pragma HLS PIPELINE II=1
            streamOut.Push(tensor[iter]);
        }
    }

    // The conditional should be outside the pipelined loop, otherwise there wont be any burst accesses.
    if(destBank == bankIndex){
        #ifdef KERNEL_LOGS
            cout<<"DataMoverV2_UnitBank"<<bankIndex<<": Writing "<<vecCount<<" vectors..."<<endl;
        #endif
            
        LoopWrite:
        for(unsigned iter=0; iter<vecCount; iter++){
            #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            #pragma HLS PIPELINE II=1
            tensor[iter] = streamIn.Pop();
        }
    }
}

#ifdef USEMEMORYBANK0
void DataMoverV2_UnitBank0(
        MemoryPackF_t *tensor,
        Stream<MemoryPackF_t, PipeDepth> &streamIn,
        Stream<MemoryPackF_t, PipeDepth> &streamOut,
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
    // Because hlslib cannot handle template functions in dataflow scheme.
    DataMoverV2_UnitBankX<0>(tensor, streamIn, streamOut, srcBank, destBank, vecCount);
}
#endif

#ifdef USEMEMORYBANK1
void DataMoverV2_UnitBank1(
        MemoryPackF_t *tensor,
        Stream<MemoryPackF_t, PipeDepth> &streamIn,
        Stream<MemoryPackF_t, PipeDepth> &streamOut,
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
    // Because hlslib cannot handle template functions in dataflow scheme.
    DataMoverV2_UnitBankX<1>(tensor, streamIn, streamOut, srcBank, destBank, vecCount);
}
#endif

#ifdef USEMEMORYBANK2
void DataMoverV2_UnitBank2(
        MemoryPackF_t *tensor,
        Stream<MemoryPackF_t, PipeDepth> &streamIn,
        Stream<MemoryPackF_t, PipeDepth> &streamOut,
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
    // Because hlslib cannot handle template functions in dataflow scheme.
    DataMoverV2_UnitBankX<2>(tensor, streamIn, streamOut, srcBank, destBank, vecCount);
}
#endif

#ifdef USEMEMORYBANK3
void DataMoverV2_UnitBank3(
        MemoryPackF_t *tensor,
        Stream<MemoryPackF_t, PipeDepth> &streamIn,
        Stream<MemoryPackF_t, PipeDepth> &streamOut,
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
    // Because hlslib cannot handle template functions in dataflow scheme.
    DataMoverV2_UnitBankX<3>(tensor, streamIn, streamOut, srcBank, destBank, vecCount);
}
#endif

void DataMoverV2_UnitBroker(
#ifdef USEMEMORYBANK0
        Stream<MemoryPackF_t, PipeDepth> &streamU0In,  // Should only be pushed
        Stream<MemoryPackF_t, PipeDepth> &streamU0Out, // Should only be popped
#endif
#ifdef USEMEMORYBANK1 
        Stream<MemoryPackF_t, PipeDepth> &streamU1In,  // Should only be pushed
        Stream<MemoryPackF_t, PipeDepth> &streamU1Out, // Should only be popped
#endif
#ifdef USEMEMORYBANK2     
        Stream<MemoryPackF_t, PipeDepth> &streamU2In,  // Should only be pushed
        Stream<MemoryPackF_t, PipeDepth> &streamU2Out, // Should only be popped
#endif
#ifdef USEMEMORYBANK3 
        Stream<MemoryPackF_t, PipeDepth> &streamU3In,  // Should only be pushed
        Stream<MemoryPackF_t, PipeDepth> &streamU3Out, // Should only be popped
#endif 
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){

    LoopIter:
    for(unsigned iter=0; iter<vecCount; iter++){
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
        #pragma HLS PIPELINE II=1

        MemoryPackF_t vec;
        // 1. Get the vector
#ifdef USEMEMORYBANK0 
        if(srcBank==0){
            vec = streamU0Out.Pop();
        }
#endif
#ifdef USEMEMORYBANK1 
        if(srcBank==1){
            vec = streamU1Out.Pop();
        }
#endif
#ifdef USEMEMORYBANK2 
        if(srcBank==2){
            vec = streamU2Out.Pop();
        }
#endif
#ifdef USEMEMORYBANK3 
        if(srcBank==3){
            vec = streamU3Out.Pop();
        }
#endif

        // 2. Send it
#ifdef USEMEMORYBANK0 
        if(destBank==0){
            streamU0In.Push(vec);
        }
#endif
#ifdef USEMEMORYBANK1 
        if(destBank==1){
            streamU1In.Push(vec);
        }
#endif
#ifdef USEMEMORYBANK2 
        if(destBank==2){
            streamU2In.Push(vec);
        }
#endif
#ifdef USEMEMORYBANK3 
        if(destBank==3){
            streamU3In.Push(vec);
        }
#endif
    }

}


void DataMoverV2(
#ifdef USEMEMORYBANK0
        MemoryPackF_t *dataBank0,
#endif
#ifdef USEMEMORYBANK1        
        MemoryPackF_t *dataBank1,
#endif
#ifdef USEMEMORYBANK2
        MemoryPackF_t *dataBank2,
#endif
#ifdef USEMEMORYBANK3        
        MemoryPackF_t *dataBank3,
#endif  
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){

#pragma HLS INLINE

    /*
     * UnitBank0 ========= |-------------|
     *                     |             | 
     *                     |             | 
     * UnitBank1 ========= |             |
     *                     |   Unit      | 
     *                     |     Broker  | 
     * UnitBank2 ========= |             |
     *                     |             | 
     *                     |             | 
     * UnitBank3 ========= |-------------|
     *
     */

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
   
#ifndef USEMEMORYBANK0 
    assert(srcBank!=0 && destBank!=0);
#endif
#ifndef USEMEMORYBANK1 
    assert(srcBank!=1 && destBank!=1);
#endif
#ifndef USEMEMORYBANK2 
    assert(srcBank!=2 && destBank!=2);
#endif
#ifndef USEMEMORYBANK3
    assert(srcBank!=3 && destBank!=3);
#endif

#ifdef USEMEMORYBANK0 
    Stream<MemoryPackF_t, PipeDepth> strmU0in;
    Stream<MemoryPackF_t, PipeDepth> strmU0out;
    #pragma HLS STREAM variable=strmU0in depth=PipeDepth
    #pragma HLS STREAM variable=strmU0out depth=PipeDepth
#endif
#ifdef USEMEMORYBANK1  
    Stream<MemoryPackF_t, PipeDepth> strmU1in;
    Stream<MemoryPackF_t, PipeDepth> strmU1out;
    #pragma HLS STREAM variable=strmU1in depth=PipeDepth
    #pragma HLS STREAM variable=strmU1out depth=PipeDepth
#endif
#ifdef USEMEMORYBANK2  
    Stream<MemoryPackF_t, PipeDepth> strmU2in;
    Stream<MemoryPackF_t, PipeDepth> strmU2out;
    #pragma HLS STREAM variable=strmU2in depth=PipeDepth
    #pragma HLS STREAM variable=strmU2out depth=PipeDepth
#endif
#ifdef USEMEMORYBANK3  
    Stream<MemoryPackF_t, PipeDepth> strmU3in;
    Stream<MemoryPackF_t, PipeDepth> strmU3out;
    #pragma HLS STREAM variable=strmU3in depth=PipeDepth
    #pragma HLS STREAM variable=strmU3out depth=PipeDepth
#endif

    #pragma HLS DATAFLOW

    HLSLIB_DATAFLOW_INIT();

#ifdef USEMEMORYBANK0  
    HLSLIB_DATAFLOW_FUNCTION(DataMoverV2_UnitBank0, dataBank0, strmU0in, strmU0out, srcBank, destBank, vecCount);
#endif
#ifdef USEMEMORYBANK1  
    HLSLIB_DATAFLOW_FUNCTION(DataMoverV2_UnitBank1, dataBank1, strmU1in, strmU1out, srcBank, destBank, vecCount);
#endif
#ifdef USEMEMORYBANK2  
    HLSLIB_DATAFLOW_FUNCTION(DataMoverV2_UnitBank2, dataBank2, strmU2in, strmU2out, srcBank, destBank, vecCount);
#endif
#ifdef USEMEMORYBANK3  
    HLSLIB_DATAFLOW_FUNCTION(DataMoverV2_UnitBank3, dataBank3, strmU3in, strmU3out, srcBank, destBank, vecCount);
#endif

    HLSLIB_DATAFLOW_FUNCTION(DataMoverV2_UnitBroker, 
#ifdef USEMEMORYBANK0
        strmU0in,  
        strmU0out, 
#endif
#ifdef USEMEMORYBANK1 
        strmU1in, 
        strmU1out, 
#endif
#ifdef USEMEMORYBANK2     
        strmU2in,  
        strmU2out, 
#endif
#ifdef USEMEMORYBANK3 
        strmU3in, 
        strmU3out, 
#endif    
        srcBank,
        destBank,
        vecCount);

    HLSLIB_DATAFLOW_FINALIZE();
}


/*
void _DataMoverV1(
#ifdef USEMEMORYBANK0
        MemoryPackF_t *dataBank0,
#endif
#ifdef USEMEMORYBANK1        
        MemoryPackF_t *dataBank1,
#endif
#ifdef USEMEMORYBANK2
        MemoryPackF_t *dataBank2,
#endif
#ifdef USEMEMORYBANK3        
        MemoryPackF_t *dataBank3,
#endif  
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
    
#pragma HLS INLINE
   
#ifndef USEMEMORYBANK0 
    assert(srcBank!=0 && destBank!=0);
#endif
#ifndef USEMEMORYBANK1 
    assert(srcBank!=1 && destBank!=1);
#endif
#ifndef USEMEMORYBANK2 
    assert(srcBank!=2 && destBank!=2);
#endif
#ifndef USEMEMORYBANK3
    assert(srcBank!=3 && destBank!=3);
#endif
    LoopIter:
    for(unsigned iter=0; iter<vecCount; iter++){
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
#pragma HLS PIPELINE II=1

#ifdef USEMEMORYBANK0    
        const MemoryPackF_t buff0 = dataBank0[iter];
#endif  
#ifdef USEMEMORYBANK1        
        const MemoryPackF_t buff1 = dataBank1[iter];
#endif  
#ifdef USEMEMORYBANK2  
        const MemoryPackF_t buff2 = dataBank2[iter];
#endif  
#ifdef USEMEMORYBANK3  
        const MemoryPackF_t buff3 = dataBank3[iter];
#endif  

        //---------------------------------------------------------
        
        if(destBank==0){
#ifdef USEMEMORYBANK0  
            dataBank0[iter] = 
#ifdef USEMEMORYBANK3
                (srcBank==3)? buff3:
#endif
#ifdef USEMEMORYBANK2
                (srcBank==2)? buff2:
#endif
#ifdef USEMEMORYBANK1
                (srcBank==1)? buff1:
#endif
                buff0;
#endif  

        }else if(destBank==1){
#ifdef USEMEMORYBANK1  
            dataBank1[iter] = 
#ifdef USEMEMORYBANK3
                (srcBank==3)? buff3:
#endif
#ifdef USEMEMORYBANK2
                (srcBank==2)? buff2:
#endif
#ifdef USEMEMORYBANK0
                (srcBank==0)? buff0:
#endif
                buff1;
#endif  
        }else if(destBank==2){
#ifdef USEMEMORYBANK2  
            dataBank2[iter] = 
#ifdef USEMEMORYBANK3
                (srcBank==3)? buff3:
#endif
#ifdef USEMEMORYBANK1
                (srcBank==1)? buff1:
#endif
#ifdef USEMEMORYBANK0
                (srcBank==0)? buff0:
#endif
                buff2;
#endif  
        }else if(destBank==3){
#ifdef USEMEMORYBANK3  
            dataBank3[iter] = 
#ifdef USEMEMORYBANK2
                (srcBank==2)? buff2:
#endif
#ifdef USEMEMORYBANK1
                (srcBank==1)? buff1:
#endif
#ifdef USEMEMORYBANK0
                (srcBank==0)? buff0:
#endif
                buff3;
#endif  
        }else{
            assert(0);
        }
    }
}
*/

extern "C" {
void task_datamover(
#ifdef USEMEMORYBANK0
        MemoryPackF_t *dataBank0,
#endif
#ifdef USEMEMORYBANK1
        MemoryPackF_t *dataBank1,
#endif
#ifdef USEMEMORYBANK2
        MemoryPackF_t *dataBank2,
#endif
#ifdef USEMEMORYBANK3
        MemoryPackF_t *dataBank3,
#endif
        const unsigned srcBank,
        const unsigned destBank,
        const unsigned vecCount){
#ifdef USEMEMORYBANK0
    #pragma HLS INTERFACE m_axi port=dataBank0 offset=slave bundle=gmem1 max_read_burst_length=32 max_write_burst_length=32
    #pragma HLS INTERFACE s_axilite port=dataBank0 bundle=control
#endif
#ifdef USEMEMORYBANK1
    #pragma HLS INTERFACE m_axi port=dataBank1 offset=slave bundle=gmem2 max_read_burst_length=32 max_write_burst_length=32
    #pragma HLS INTERFACE s_axilite port=dataBank1 bundle=control
#endif
#ifdef USEMEMORYBANK2
    #pragma HLS INTERFACE m_axi port=dataBank2 offset=slave bundle=gmem3 max_read_burst_length=32 max_write_burst_length=32
    #pragma HLS INTERFACE s_axilite port=dataBank2 bundle=control
#endif
#ifdef USEMEMORYBANK3
    #pragma HLS INTERFACE m_axi port=dataBank3 offset=slave bundle=gmem4 max_read_burst_length=32 max_write_burst_length=32
    #pragma HLS INTERFACE s_axilite port=dataBank3 bundle=control
#endif
#pragma HLS INTERFACE s_axilite port=srcBank bundle=control
#pragma HLS INTERFACE s_axilite port=destBank bundle=control
#pragma HLS INTERFACE s_axilite port=vecCount bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // For this kernel to be usable for OclTensorI, size of the data types should match. 
    assert(CONFIG_DTYPE_SIZE == sizeof(unsigned));

    DataMoverV2(
#ifdef USEMEMORYBANK0
        dataBank0,
#endif
#ifdef USEMEMORYBANK1
        dataBank1,
#endif
#ifdef USEMEMORYBANK2
        dataBank2,
#endif
#ifdef USEMEMORYBANK3
        dataBank3,
#endif
        srcBank,
        destBank,
        vecCount);
}
}

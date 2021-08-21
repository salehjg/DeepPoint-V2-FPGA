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
using namespace ConfigTaskTile; 

/**
 * @brief      Performs tile operation on the input tensor of rank 2 over the last axis 2.
 *             This kernel complies with the padded last dim policy:
 *               1) The input tensor of rank 2 should be padded over the last dimension(dim1).
 *               2) The output tensor of rank 3 will be padded over the last dimension(tileSize).
 *             The latency will be reported for an input tensor of shape 5x1024 and tileSize of 1024 with axi512.
 *             This kernel supports burst read/write.
 *
 * @param[in]  inputTn   The input tn of shape dim0*dim1
 * @param      outputTn  The output tn of shape dim0*dim1*tileSize
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  tileSize  The tile size
 */
void TileRank2Axis2(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,   
    const unsigned tileSize){

    #pragma HLS INLINE

    // input: BxN, tileAxis=2 ===> output: BxNxT ===> lastDim: T,(tileSize)

    CONFIG_DTYPE buff[MaxSliceLen];
#pragma HLS ARRAY_PARTITION variable=buff cyclic factor=CONFIG_M_AXI_WIDTH dim=1

    const unsigned dim1Padded = MakeDivisible<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSliceIn = dim1Padded/CONFIG_M_AXI_WIDTH;

    const unsigned tileSizePadded = MakeDivisible<unsigned>(tileSize, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSliceOut = tileSizePadded/CONFIG_M_AXI_WIDTH;

#ifdef KERNEL_LOGS
    cout<<"dim0: "<<dim0<<endl;
    cout<<"dim1: "<<dim1<<endl;
    cout<<"tileSize: "<<tileSize<<endl;
    cout<<"dim1Padded: "<<dim1Padded<<endl;
    cout<<"vecsPerSliceIn: "<<vecsPerSliceIn<<endl;
    cout<<"tileSizePadded: "<<tileSizePadded<<endl;
    cout<<"vecsPerSliceOut: "<<vecsPerSliceOut<<endl;
#endif

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
		#pragma HLS LOOP_TRIPCOUNT min=5 max=5

        LoopDim1:
        for(unsigned id1=0; id1<vecsPerSliceIn; id1++){
			#pragma HLS LOOP_TRIPCOUNT min=64 max=64
            #pragma HLS PIPELINE II=1
            const unsigned indxS = d0*vecsPerSliceIn + id1;
            MemoryPackF_t vec = inputTn[indxS];
            LoopFill1Unrolled:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                const unsigned indxL = id1*CONFIG_M_AXI_WIDTH + i;
                buff[indxL] = vec[i];
            }
        }

        LoopDim1O:
        for(unsigned d1=0; d1<dim1; d1++){
			#pragma HLS LOOP_TRIPCOUNT min=64 max=64
            LoopTile:
            for(unsigned id2=0; id2<vecsPerSliceOut; id2++) {
				#pragma HLS LOOP_TRIPCOUNT min=64 max=64
                #pragma HLS PIPELINE II=1
                MemoryPackF_t vec;
                const CONFIG_DTYPE val = buff[d1];
                LoopFill2Unrolled:
                for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
                    #pragma HLS UNROLL
                    vec[i] = (((id2*CONFIG_M_AXI_WIDTH + i) < tileSize) ? val : 0);
                }
                const unsigned indxD = d0*dim1*vecsPerSliceOut + d1*vecsPerSliceOut + id2;
                outputTn[indxD] = vec;
            }
        }

    }
}

/**
 * @brief      Performs tile operation on the input tensor of rank 2 over the last axis 1.
 *             This kernel complies with the padded last dim policy.
 *               1) The input tensor of rank 2 should be padded over the last dimension(dim1).
 *               2) The output tensor of rank 3 will be padded over the last dimension(dim1).
 *             The latency will be reported for an input tensor of shape 5x1024 and tileSize of 1024 with axi512.
 *             This kernel supports burst read/write.
 *
 * @param[in]  inputTn   The input tn of shape dim0*dim1
 * @param      outputTn  The output tn of shape dim0*tileSize*dim1
 * @param[in]  dim0      The dim 0
 * @param[in]  dim1      The dim 1
 * @param[in]  tileSize  The tile size
 */
void TileRank2Axis1(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned tileSize){

    #pragma HLS INLINE
    
    // input: BxN, tileAxis=1 ===> output: BxTxN ===> lastDim: N,(dim1)

    CONFIG_DTYPE buff[MaxSliceLen];
#pragma HLS ARRAY_PARTITION variable=buff cyclic factor=CONFIG_M_AXI_WIDTH dim=1

    const unsigned dim1Padded = MakeDivisible<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSliceIn = dim1Padded/CONFIG_M_AXI_WIDTH;
    const unsigned vecsPerSliceOut = vecsPerSliceIn;

#ifdef KERNEL_LOGS
    cout<<"dim0: "<<dim0<<endl;
    cout<<"dim1: "<<dim1<<endl;
    cout<<"tileSize: "<<tileSize<<endl;
    cout<<"dim1Padded: "<<dim1Padded<<endl;
    cout<<"vecsPerSliceIn: "<<vecsPerSliceIn<<endl;
    cout<<"vecsPerSliceOut: "<<vecsPerSliceOut<<endl;
#endif

    LoopDim0:
    for(unsigned d0=0; d0<dim0; d0++){
		#pragma HLS LOOP_TRIPCOUNT min=5 max=5
        LoopDim1:
        for(unsigned id1=0; id1<vecsPerSliceIn; id1++){
			#pragma HLS LOOP_TRIPCOUNT min=64 max=64
            #pragma HLS PIPELINE II=1
            const unsigned indxS = d0*vecsPerSliceIn + id1;
            MemoryPackF_t vec = inputTn[indxS];
            LoopFill1Unrolled:
            for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                const unsigned indxL = id1*CONFIG_M_AXI_WIDTH + i;
                buff[indxL] = vec[i];
            }
        }
        LoopDim1O:
		for(unsigned d1=0; d1<tileSize; d1++){
			#pragma HLS LOOP_TRIPCOUNT min=64 max=64
			LoopTile:
			for(unsigned id2=0; id2<vecsPerSliceOut; id2++) {
				#pragma HLS LOOP_TRIPCOUNT min=64 max=64
				#pragma HLS PIPELINE II=1
				MemoryPackF_t vec;
				LoopFill2Unrolled:
				for (unsigned i = 0; i < CONFIG_M_AXI_WIDTH; i++) {
					#pragma HLS UNROLL
					const unsigned indxL = id2*CONFIG_M_AXI_WIDTH + i;
					vec[i] = buff[indxL];
				}
				const unsigned indxD = d0*tileSize*vecsPerSliceOut + d1*vecsPerSliceOut + id2;
				outputTn[indxD] = vec;
			}
		}
    }
}

extern "C" {
void task_tile(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2, 
        const unsigned rank,
        const unsigned tileAxis,
        const unsigned tileSize){
#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=16
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=dim2 bundle=control
#pragma HLS INTERFACE s_axilite port=rank bundle=control
#pragma HLS INTERFACE s_axilite port=tileAxis bundle=control
#pragma HLS INTERFACE s_axilite port=tileSize bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    assert(rank==2 || rank==3);
    assert(
        (rank==2 && tileAxis==1)||
        (rank==2 && tileAxis==2)||
        (rank==3 && tileAxis==2)
        );

    if((rank==2 && tileAxis==1) || (rank==3 && tileAxis==2)){
        //input: BxN ===> output: BxTxN ===> lastDim: N,(dim1)
        TileRank2Axis1(
            inputTn, 
            outputTn, 
            (rank==2 && tileAxis==1) ? dim0 : dim0*dim1, 
            (rank==2 && tileAxis==1) ? dim1 : dim2, 
            tileSize);
#ifdef KERNEL_LOGS
        cout<<"Selected SubFunc: TileRank2Axis1(1 or 3)"<<endl;
#endif
    }else if(rank==2 && tileAxis==2){
        //input: BxN ===> output: BxNxT ===> lastDim: T,(tileSize)
        TileRank2Axis2(inputTn, outputTn, dim0, dim1, tileSize);
#ifdef KERNEL_LOGS
        cout<<"Selected SubFunc: TileRank2Axis2(2)"<<endl;
#endif
    }
}
}

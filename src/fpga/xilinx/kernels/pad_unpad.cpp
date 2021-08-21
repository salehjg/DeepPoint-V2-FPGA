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

/**
 * @brief      Pads the last dimension of the given tensor of rank two to the given value(dim1Padded) 
 *             Tensor of higher ranks should be flattened into two virtual dimensions for this kernel to be usable.
 *             Sub-vec padding is disabled, as Cpu last dim padding is decided to be enabled.
 *
 * @param      inputTn     The input tensor
 * @param      outputTn    The output tensor
 * @param[in]  dim0        The dim 0
 * @param[in]  dim1        The dim 1 (original shape)
 * @param[in]  dim1Padded  The target value for dim 1 after padding
 * @param[in]  lcm         The least common multiple of dim1 and m_axi width(in words for ex 16 or so)(don't care for dim1>=m_axi_width)
 */
void PadLastDimSuperVec(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Padded){

    #pragma HLS INLINE

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif


    assert(dim1>=CONFIG_M_AXI_WIDTH);
    assert(dim1%CONFIG_M_AXI_WIDTH==0);
    
    assert(dim1Padded>=CONFIG_M_AXI_WIDTH);
    assert(dim1Padded%CONFIG_M_AXI_WIDTH==0);
    assert(dim1Padded>dim1);

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(outputTn->kWidth==CONFIG_M_AXI_WIDTH);

    const auto idim1 = dim1/CONFIG_M_AXI_WIDTH;
    const auto idim1Padded = dim1Padded/CONFIG_M_AXI_WIDTH;
    unsigned int indxS, indxD;


#ifdef KERNEL_LOGS
    cout<<"idim1: "<<idim1<<endl;
    cout<<"idim1Padded: "<<idim1Padded<<endl;
#endif

    LoopD0:
    for(unsigned int d0=0; d0<dim0; d0++){
        LoopD1:
        for(unsigned int id1=0; id1<idim1; id1++){
            #pragma HLS PIPELINE II=1
            indxS = d0*idim1+id1;
            indxD = d0*idim1Padded+id1;
#ifdef KERNEL_LOGS
            cout<<"## d0: "<<d0<<" id1: "<<id1<<" indxS: "<<indxS<<" indxD: "<<indxD<<endl;
#endif
            MemoryPackF_t tmpVec1 = inputTn[indxS];
            outputTn[indxD] = tmpVec1;
        }
    }
}

/*
// Tested Kernel - Works Perfectly
// DISABLED, Reason: Cpu last dim padding is enabled and all of
//                   the kernels should consider reading and writing in 
//                   a last dim padded manner. 
void PadLastDimSubVec(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Padded,
    const unsigned int lcm){

    // Sub Vec Padding Kernel

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    //assert(dim1<CONFIG_M_AXI_WIDTH); //XOCC crashes when this line is uncommented.

    assert(dim1Padded%CONFIG_M_AXI_WIDTH==0);
    assert(dim1Padded>dim1);
    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(outputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(lcm<=LOCAL_BUFF_LEN);

    const unsigned int bunchVecCount = lcm/CONFIG_M_AXI_WIDTH;
    const unsigned int bunchSliceCount = lcm/dim1;
    const unsigned int vecPerOutputSlice = dim1Padded/CONFIG_M_AXI_WIDTH;
    const auto limitS=DivCeil<unsigned int>(dim0*dim1, CONFIG_M_AXI_WIDTH);
    const auto limitD=DivCeil<unsigned int>(dim0*dim1Padded, CONFIG_M_AXI_WIDTH);
    unsigned int indxS, indxD, indxL1, indxL2;

    CONFIG_DTYPE buff[LOCAL_BUFF_LEN];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=buff cyclic factor=CONFIG_M_AXI_WIDTH dim=1)

    MemoryPackF_t tmpVec1, tmpVec2;
    const auto bunchCount = DivCeil<unsigned int>(dim0*dim1, lcm);
#ifdef KERNEL_LOGS
    cout<<"limitS: "<<limitS<<endl;
    cout<<"limitD: "<<limitD<<endl;
#endif
    LoopIter0:
    for(unsigned int iter=0; 
        iter<bunchCount;
        iter++){
        LoopDim0Init:
        for(unsigned int id0=0; id0<bunchVecCount; id0++){
            #pragma HLS PIPELINE II=1
#ifdef KERNEL_LOGS
            cout<<"## iter: " << iter << " id0: "<<id0<<endl;
#endif       
            // id0 is the local index of vector, NOT the index of slice in dim0.
            
            ///TODO: check limits for indxS 
            indxS = iter*bunchVecCount+id0;
            if(indxS<limitS){
#ifdef KERNEL_LOGS
                cout<<"*indxS: "<<indxS<<endl;
#endif
                tmpVec1 = inputTn[indxS];
            }else{
                for(unsigned int i=0; i<CONFIG_M_AXI_WIDTH; i++){
                    #pragma HLS UNROLL
                    tmpVec1[i]=0;
                }
#ifdef KERNEL_LOGS
                cout<<"limitS is hit *indxS: "<<indxS<<endl;
#endif
            }

            LoopUnrol0:
            for(unsigned int i=0; i<CONFIG_M_AXI_WIDTH; i++){
                #pragma HLS UNROLL
                indxL1=id0*CONFIG_M_AXI_WIDTH+i;
#ifdef KERNEL_LOGS
                cout<<"--indxL1: "<<indxL1<<endl;
#endif
                buff[indxL1] = tmpVec1[i];
            }

        }

        LoopIter1:
        for(unsigned int iter1=0;
            iter1<bunchSliceCount;
            iter1++){
            #pragma HLS PIPELINE II=1
#ifdef KERNEL_LOGS
            cout<<"## iter: " << iter << " iter1: "<<iter1<<endl;
#endif
            // Because we have "dim1<CONFIG_M_AXI_WIDTH", we can ignore the need 
            // for a "for-loop" of:
            // for(d1=0; d1<dim1Padded/CONFIG_M_AXI_WIDTH; d1++) 
            // and just put the slice in the first index(d1=0)
            //for(unsigned int d1=0; d1<dim1Padded/CONFIG_M_AXI_WIDTH; d1++)
            unsigned int d1=0;
            {
                if(d1==0){
                    LoopUnrol1:
                    for(unsigned int i=0; i<CONFIG_M_AXI_WIDTH; i++){
                        #pragma HLS UNROLL
                        if(i<dim1){
                            indxL2=iter1*dim1+i;
#ifdef KERNEL_LOGS
                            cout<<"==indxL2: "<<indxL2<<endl;
#endif
                            tmpVec2[i] = buff[indxL2];
                        }else{
                            tmpVec2[i] = 0;
                        }
                    }
                }

                indxD = iter*(bunchSliceCount*vecPerOutputSlice)+
                        iter1*vecPerOutputSlice+
                        d1;
                if(indxD<limitD){
#ifdef KERNEL_LOGS
                    cout<<"**indxD: "<<indxD<<endl;
#endif
                    outputTn[indxD] = tmpVec2;
                }else{
#ifdef KERNEL_LOGS
                    cout<<"limitD is hit **indxD: "<<indxD<<endl;
#endif
                }
            }
        }

    }
}
*/

/**
 * @brief      Unpads the padded input tensor on the last dimension.
 *             Currently 
 *                1)The input tensor's last dimension should be greater than m_axi_width and
 *                  should be divisible by m_axi_width.
 *                2)The same conditions as (1) are applied to 'dim1Unpadded'
 *
 * @param[in]  inputTn       The input tn
 * @param      outputTn      The output tn
 * @param[in]  dim0          The dim 0
 * @param[in]  dim1          The dim 1
 * @param[in]  dim1Unpadded  The dim 1 unpadded
 */
void UnpadLastDimSuperVec(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned int dim0,
    const unsigned int dim1,
    const unsigned int dim1Unpadded){

    #pragma HLS INLINE
    
#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif
    
    //assert(dim1>=CONFIG_M_AXI_WIDTH); //XOCC crashes when this line is uncommented.
    assert(dim1%CONFIG_M_AXI_WIDTH==0);
    
    assert(dim1Unpadded>=CONFIG_M_AXI_WIDTH);
    assert(dim1Unpadded%CONFIG_M_AXI_WIDTH==0);
    assert(dim1Unpadded<dim1);

    assert(inputTn->kWidth==CONFIG_M_AXI_WIDTH);
    assert(outputTn->kWidth==CONFIG_M_AXI_WIDTH);

    const auto idim1 = dim1/CONFIG_M_AXI_WIDTH;
    const auto idim1Unpadded = dim1Unpadded/CONFIG_M_AXI_WIDTH;
    unsigned int indxS, indxD;

#ifdef KERNEL_LOGS
    cout<<"idim1: "<<idim1<<endl;
    cout<<"idim1Unpadded: "<<idim1Unpadded<<endl;
#endif

    LoopD0:
    for(unsigned int d0=0; d0<dim0; d0++){
        LoopD1:
        for(unsigned int id1=0; id1<idim1Unpadded; id1++){
            #pragma HLS PIPELINE II=1
            indxS = d0*idim1+id1;
            indxD = d0*idim1Unpadded+id1;
#ifdef KERNEL_LOGS
            cout<<"## d0: "<<d0<<" id1: "<<id1<<" indxS: "<<indxS<<" indxD: "<<indxD<<endl;
#endif
            MemoryPackF_t tmpVec1 = inputTn[indxS];
            outputTn[indxD] = tmpVec1;
        }
    }
}

extern "C" {
void task_pad_unpad(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned mode,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned pad_dim1Padded,
    const unsigned pad_lcm,
    const unsigned unpad_dim1Unpadded){

#pragma HLS INTERFACE m_axi port=inputTn offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=16
#pragma HLS INTERFACE m_axi port=outputTn offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn bundle=control

#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control

#pragma HLS INTERFACE s_axilite port=pad_dim1Padded bundle=control
#pragma HLS INTERFACE s_axilite port=pad_lcm bundle=control
#pragma HLS INTERFACE s_axilite port=unpad_dim1Unpadded bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#ifdef KERNEL_LOGS
    cout<<"Simulation mode is enabled."<<endl;
#endif

    if(mode==1){ //PAD
        if(dim1*2/2<CONFIG_M_AXI_WIDTH){ // Without "*2/2", XOCC crashes in internal "RangeAnalysis" method.
            /*
            PadLastDimSubVec(
                inputTn,
                outputTn,
                dim0,
                dim1,
                pad_dim1Padded,
                pad_lcm);
                */
        }else{
            PadLastDimSuperVec(
                inputTn,
                outputTn,
                dim0,
                dim1,
                pad_dim1Padded);
        }
    }

    if(mode==2){ //UNPAD
        if(dim1>CONFIG_M_AXI_WIDTH){
            UnpadLastDimSuperVec(inputTn, outputTn, dim0, dim1, unpad_dim1Unpadded);
        }     
    }

}
}

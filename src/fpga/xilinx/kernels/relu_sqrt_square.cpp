#include <cassert>
#include <iostream>
#include <limits>
#include <hls_math.h>
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Operators.h"
#include "hlslib/xilinx/TreeReduce.h"
#include "AxiHelper.h"
#include "xilinx/config.h"

using namespace std; 
using namespace ConfigTaskReluSqrtSquare;

void Relu_V1(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned len){

    #pragma HLS INLINE

	for(unsigned iVec=0; iVec<len; iVec++){
		#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
		#pragma HLS PIPELINE II=1
		MemoryPackF_t vec = inputTn[iVec];
		LoopProcessUnrolled:
		for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
			#pragma HLS UNROLL
			const CONFIG_DTYPE val = vec[i];
			vec[i] = (val>0)?val:0;
		}
		outputTn[iVec] = vec;

	}
}

void Sqrt_V1(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned len){

	#pragma HLS INLINE

	for(unsigned iVec=0; iVec<len; iVec++){
		#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
		#pragma HLS PIPELINE II=1
		MemoryPackF_t vec = inputTn[iVec];
		LoopProcessUnrolled:
		for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
			#pragma HLS UNROLL
			const CONFIG_DTYPE val = vec[i];
			vec[i] = sqrt(val);
		}
		outputTn[iVec] = vec;

	}
}

void Square_V1(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned len){

	#pragma HLS INLINE
	
	for(unsigned iVec=0; iVec<len; iVec++){
		#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
		#pragma HLS PIPELINE II=1
		MemoryPackF_t vec = inputTn[iVec];
		LoopProcessUnrolled:
		for(unsigned i=0; i<CONFIG_M_AXI_WIDTH; i++){
			#pragma HLS UNROLL
			const CONFIG_DTYPE val = vec[i];
			vec[i] = val*val;
		}
		outputTn[iVec] = vec;

	}
}

extern "C" {

/**
 * @brief      This kernel supports burst read/write.
 *
 * @param[in]  inputTn   The input tn
 * @param      outputTn  The output tn
 * @param[in]  len       The length
 * @param[in]  mode      The mode
 */
void task_relu_sqrt_square(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned len,
        const unsigned mode){
#pragma HLS INTERFACE m_axi     port=inputTn    offset=slave bundle=gmem1 max_read_burst_length=16 max_write_burst_length=16
#pragma HLS INTERFACE m_axi     port=outputTn   offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=inputTn    bundle=control
#pragma HLS INTERFACE s_axilite port=outputTn   bundle=control
#pragma HLS INTERFACE s_axilite port=len        bundle=control
#pragma HLS INTERFACE s_axilite port=mode        bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control
	if(mode==ModeRelu){
		Relu_V1(inputTn, outputTn, len);
	}else if(mode==ModeSqrt){
		Sqrt_V1(inputTn, outputTn, len);
	}else if(mode==ModeSquare){
		Square_V1(inputTn, outputTn, len);
	}
}
}

#include <algorithm>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include "Utility.h"
#include "AxiHelper.h"
#include "PaddingCpu.h"

using namespace std;
using namespace ConfigTaskTranspose;

extern "C"
void task_transpose(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2);

void GoldTranspose(
    const CONFIG_DTYPE *inputTn,
    CONFIG_DTYPE *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2){

    unsigned indxS, indxD;

    for(int b=0;b<dim0;b++){
        for (int j = 0; j < dim1; j++) {
            for (int i = 0; i < dim2 ; i++) {
                indxS = b * dim1 * dim2 + j * dim2 + i;
                indxD = b * dim1 * dim2 + i * dim1 + j;
                outputTn[indxD] = inputTn[indxS];
            }
        }
    }
}

template<unsigned vecSize>
int TestTranspose(
    const string& testName,
    const std::vector<unsigned> shape){

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;
    const unsigned rank = shape.size();
    const unsigned dim2 = shape[rank-1]; 
    const unsigned dim2Padded = MakeDivisible<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    
    unsigned dim0, dim1;
    if(rank==2){
        dim0=1;
        dim1=shape[0];
    }else{
        dim0=shape[0];
        dim1=shape[1];
    }

    const unsigned dim1Padded = MakeDivisible<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    unsigned lenInput = dim0*dim1*dim2;
    unsigned lenInputPadded = dim0*dim1*dim2Padded;
    unsigned lenOutputUdt = dim0*dim2*dim1Padded;

    const unsigned lenOutput = lenInput;
    
    std::vector<CONFIG_DTYPE> hostInputTn(lenInput); 
    std::vector<CONFIG_DTYPE> hostInputTnPadded(lenInputPadded);
    std::vector<CONFIG_DTYPE> hostGold(lenOutput);
    std::vector<CONFIG_DTYPE> hostUdtPadded(lenOutputUdt);
    std::vector<CONFIG_DTYPE> hostUdtUnpadded(lenOutput);

    /*
    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTn.begin(), hostInputTn.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });*/

    for(unsigned idx = 0; idx < lenInput; idx++){
        hostInputTn[idx] = idx;
    }

    PadTensor<CONFIG_DTYPE>(hostInputTn, hostInputTnPadded, lenInput/dim2, dim2, dim2Padded);

    auto deviceInputTn = Pack<vecSize, CONFIG_DTYPE>(hostInputTnPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostUdtPadded);

    task_transpose(
            deviceInputTn.data(),
            deviceOutputTn.data(),
            dim0,dim1,dim2);

    GoldTranspose(
        hostInputTn.data(), 
        hostGold.data(), 
        dim0,dim1,dim2);

    const auto hostUdtPadded2 = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    UnpadTensor<CONFIG_DTYPE>(hostUdtPadded2, hostUdtUnpadded, lenOutput/dim1, dim1Padded, dim1);
    bool rslt = true;

    for(unsigned d0=0; d0<dim0; d0++){ 
        for(unsigned d1=0; d1<dim2; d1++){ 
            for(unsigned d2=0; d2<dim1; d2++){ 
                unsigned indxCpu = d0*dim2*dim1+ d1*dim1+ d2;
                unsigned indxUdt = indxCpu;
                CONFIG_DTYPE rCpu = hostGold[indxCpu];
                CONFIG_DTYPE rUdt = hostUdtUnpadded[indxUdt];
                CONFIG_DTYPE diff = rCpu - rUdt;
                if(abs(diff)>1e-02){
                    printf("d0= (%03d)\td1= (%03d)\td2= (%03d)\trCPU=%f,\t\t rUDT=%f\n", d0, d1, d2, rCpu, rUdt);
                    rslt=false;
                }
            }
        }
    }

    std::cout<<std::endl;

    if(rslt){
        std::cout<<"Test \""<<testName<<"\" is successfully verified."<<std::endl;
    }

    return (rslt)? 0 : 1;
}

int main(int argc, char **argv) {
    int rslt = 0;
    rslt += TestTranspose<16>("Transpose1", {5,128,128});
    rslt += TestTranspose<16>("Transpose2", {5,1024,64});
    rslt += TestTranspose<16>("Transpose3", {5,1024,3});
    return rslt;
}

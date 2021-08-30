#include <algorithm>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <string>
#include <cassert>
#include "Utility.h"
#include "AxiHelper.h"

using namespace std;
using namespace ConfigTaskTopK;

extern "C"
void task_topk(
    const MemoryPackF_t *inputTn,
    MemoryPackI_t *indicesSplitedTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned kValue,
    const unsigned vecsPerSlice,
    const unsigned vecsPerOutputSlice);

void CpuGoldTopk(
    const CONFIG_DTYPE *inputBuff,
    unsigned *indicesSplitedBuff,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned kValue){

    unsigned indxS=0;
    CONFIG_DTYPE tmp_array[dim1];
    unsigned indices[dim1];

    for(unsigned b=0; b<dim0; b++){
        for(unsigned i=0; i<dim1; i++){
            indices[i]=i;
        }

        indxS = b*dim1 + 0;
        std::copy(inputBuff+indxS, inputBuff+indxS+dim1, tmp_array);
        std::sort(indices,
                  indices+dim1,
                  [&](int i1, int i2) { return tmp_array[i1] < tmp_array[i2]; } );
        std::copy(indices, indices+kValue, indicesSplitedBuff+(b*kValue));
    }
}

template<unsigned vecSize>
int TestTopk(
    const string& testName,
    const unsigned dim0, 
    const unsigned dim1,
    const unsigned kValue){

    assert(kValue<dim1);
    assert(dim1%vecSize==0);

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;
    cout<<"InputShape: "<<dim0<<"x"<<dim1<<endl;
    cout<<"kValue: "<<kValue<<endl;
    cout<<"OutputShape: "<<dim0<<"x"<<dim1<<endl;

    const unsigned lenInput = dim0*dim1;
    const unsigned lenOutput = dim0*kValue;
    const unsigned vecsPerSlice = DivCeil<unsigned>(dim1, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerOutputSlice = DivCeil<unsigned>(kValue, CONFIG_M_AXI_WIDTH);

    // UDT outputs are padded in the last dimension to be divisible by maxi width
    const unsigned lenOutputUdt = dim0*(vecsPerOutputSlice*CONFIG_M_AXI_WIDTH);

    std::vector<CONFIG_DTYPE> hostInputTn(lenInput); 
    std::vector<unsigned> hostGold(lenOutput);
    std::vector<unsigned> hostUDT(lenOutputUdt);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTn.begin(), hostInputTn.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    auto deviceInputTn = Pack<vecSize, CONFIG_DTYPE>(hostInputTn);
    auto deviceOutputTn = Pack<vecSize, unsigned>(hostUDT);

    // The kernel writes the results in the output tensor with padding on the last dimension.
    task_topk(deviceInputTn.data(), deviceOutputTn.data(), dim0, dim1, kValue, vecsPerSlice, vecsPerOutputSlice);
    CpuGoldTopk(hostInputTn.data(), hostGold.data(), dim0, dim1, kValue);

    const auto hostOutputTn = Unpack<vecSize, unsigned>(deviceOutputTn);
    bool rslt = true;

    for(unsigned b=0;b<dim0;b++){
        for(unsigned kk=0;kk<kValue;kk++){
            // The kernel writes the results in the output tensor with padding on the last dimension.
            unsigned indxCpu = b*kValue+kk;
            unsigned indxUdt = b*(vecsPerOutputSlice*CONFIG_M_AXI_WIDTH)+kk;
            unsigned rCpu = hostGold[indxCpu];
            unsigned rUdt = hostOutputTn[indxUdt];

            if(rCpu!=rUdt){
                printf("Index(B,K)= (%02d, %02d)\t\trCPU=%04d, rUDT=%04d\t\tValue[rCPU]:%f\tValue[rUDT]:%f\n",
                       b, kk,
                       rCpu,rUdt,
                       hostInputTn[b*dim1+rCpu], hostInputTn[b*dim1+rUdt]);
                rslt=false;
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
    int rslt0 = 0;
    rslt0 += TestTopk<16>("Topk", 5, MaxSliceLen, 20);
    rslt0 += TestTopk<16>("Topk", 5*1024, MaxSliceLen, 20);
    return rslt0;
}

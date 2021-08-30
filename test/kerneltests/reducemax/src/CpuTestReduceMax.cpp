#include <algorithm>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <string>
#include <cassert>
#include "Utility.h"
#include "AxiHelper.h"
#include "PaddingCpu.h"

using namespace std;
using namespace ConfigTaskReduce::Max3D;

extern "C"
void task_reduce(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned mode,
        const unsigned pow_y,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3);

void GoldReduceMax(
    const CONFIG_DTYPE *inputTn,
    CONFIG_DTYPE *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const bool overaxis0,
    const bool overaxis1,
    const bool overaxis2){

    unsigned indxS, indxD;
    assert(!overaxis0 && overaxis1 && !overaxis2);
    for(unsigned d0=0; d0<dim0; d0++){
        for(unsigned d2=0; d2<dim2; d2++){
            CONFIG_DTYPE maxVal = -numeric_limits<CONFIG_DTYPE>::max();
            for(unsigned d1=0; d1<dim1; d1++){
                indxS = d0*dim1*dim2 + d1*dim2 + d2;
                const CONFIG_DTYPE val = inputTn[indxS];
                maxVal = (maxVal<val)?val:maxVal;
            }
            indxD = d0*dim2 + d2;
            outputTn[indxD] = maxVal;
        }
    }

}

template<unsigned vecSize>
int TestReduceMax(
    const string& testName,
    const std::vector<unsigned> shape,
    const bool overaxis0,
    const bool overaxis1,
    const bool overaxis2){

    const unsigned rank = shape.size();
    assert(rank==3);
    assert(!overaxis0 & overaxis1 & !overaxis2);

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;

    const unsigned dim0 = shape[0];
    const unsigned dim1 = shape[1];
    const unsigned dim2 = shape[2];
    const unsigned dim2Padded = MakeDivisible<unsigned>(dim2, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim2Padded/CONFIG_M_AXI_WIDTH;

    const unsigned lenInput = dim0*dim1*dim2;
    const unsigned lenInputPadded = dim0*dim1*dim2Padded;
    const unsigned lenOutput = dim0*dim2;
    const unsigned lenOutputUdt = dim0*dim2Padded;

    std::vector<CONFIG_DTYPE> hostInputTn(lenInput); 
    std::vector<CONFIG_DTYPE> hostInputTnPadded(lenInputPadded);
    std::vector<CONFIG_DTYPE> hostGold(lenOutput);
    std::vector<CONFIG_DTYPE> hostUDT(lenOutputUdt);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTn.begin(), hostInputTn.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    PadTensor<CONFIG_DTYPE>(hostInputTn, hostInputTnPadded, dim0*dim1, dim2, dim2Padded);

    auto deviceInputTn = Pack<vecSize, CONFIG_DTYPE>(hostInputTnPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostUDT);

    task_reduce(
        deviceInputTn.data(),
        deviceOutputTn.data(),
        3,
        0,
        dim0, dim1, dim2, 0);

    GoldReduceMax(
        hostInputTn.data(), 
        hostGold.data(), 
        dim0, dim1, dim2,
        overaxis0, overaxis1, overaxis2);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUDT, dim0, dim2Padded, dim2);
    bool rslt = true;

    for(unsigned d0=0; d0<dim0; d0++){ 
        for(unsigned d2=0; d2<dim2; d2++){ 
            unsigned indxCpu = d0*dim2+d2;
            unsigned indxUdt = d0*dim2+d2;
            CONFIG_DTYPE rCpu = hostGold[indxCpu];
            CONFIG_DTYPE rUdt = hostUDT[indxUdt];
            CONFIG_DTYPE diff = rCpu - rUdt;
            if(abs(diff)>1e-02){
                printf("d0= (%03d)\td2= (%03d)\trCPU=%f,\t\t rUDT=%f\n", d0, d2, rCpu, rUdt);
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
    int rslt0 = TestReduceMax<16>("ReduceMax3D_FTF", {5,1024,17}, false, true, false);
    rslt0 += TestReduceMax<16>("ReduceMax3D_FTF", {2,1024,64}, false, true, false);
    rslt0 += TestReduceMax<16>("ReduceMax3D_FTF", {2,2,50}, false, true, false);
    return rslt0;
}

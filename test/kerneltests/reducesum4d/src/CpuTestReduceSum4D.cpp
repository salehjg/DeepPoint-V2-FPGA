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
using namespace ConfigTaskReduce::Sum4D;

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

void GoldReduceSum4D(
        const CONFIG_DTYPE *inputTn,
        CONFIG_DTYPE *outputTn,
        const unsigned pow_y,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3){
    unsigned indxS, indxD;
    for (unsigned d3=0; d3 < dim3; d3++){
        CONFIG_DTYPE sum=0;
        indxD = d3;
        for (unsigned d0 = 0; d0 < dim0; d0++) {
            for (unsigned d1 = 0; d1 < dim1; d1++) {
                for (unsigned d2 = 0; d2 < dim2; d2++) {

                    indxS = d0*dim1*dim2*dim3+
                            d1*dim2*dim3+
                            d2*dim3+
                            d3;

                    sum += inputTn[indxS];
                }
            }
        }
        outputTn[indxD] = sum;
    }
}

template<unsigned vecSize>
int TestReduceSum4D(
    const string& testName,
    const std::vector<unsigned> shape, 
    const unsigned pow_y){
    const unsigned rank = shape.size();
    assert(pow_y>=1);
    assert(rank==4);

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;

    const unsigned dim0 = shape[0];
    const unsigned dim1 = shape[1];
    const unsigned dim2 = shape[2];
    const unsigned dim3 = shape[3];
    const unsigned dim3Padded = MakeDivisible<unsigned>(dim3, CONFIG_M_AXI_WIDTH);
    const unsigned vecsPerSlice = dim3Padded/CONFIG_M_AXI_WIDTH;

    const unsigned lenInput = dim0*dim1*dim2*dim3;
    const unsigned lenInputPadded = dim0*dim1*dim2*dim3Padded;
    const unsigned lenOutput = dim3;
    const unsigned lenOutputUdt = dim3Padded;

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

    PadTensor<CONFIG_DTYPE>(hostInputTn, hostInputTnPadded, dim0*dim1*dim2, dim3, dim3Padded);

    auto deviceInputTn = Pack<vecSize, CONFIG_DTYPE>(hostInputTnPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostUDT);

    task_reduce(
        deviceInputTn.data(),
        deviceOutputTn.data(),
        2,
        pow_y, 
        dim0, dim1, dim2, dim3);

    GoldReduceSum4D(
        hostInputTn.data(), 
        hostGold.data(), 
        pow_y,
        dim0, dim1, dim2, dim3);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUDT, 1, dim3Padded, dim3);
    bool rslt = true;

    for(unsigned i=0; i<dim3; i++){ 
        // The kernel writes the results in the output tensor with padding on the last dimension.
        unsigned indxCpu = i;
        unsigned indxUdt = i;
        CONFIG_DTYPE rCpu = hostGold[indxCpu];
        CONFIG_DTYPE rUdt = hostUDT[indxUdt];
        CONFIG_DTYPE diff = rCpu - rUdt;
        if(abs(diff)>1e-02){
            printf("Index(d0)= (%03d)\trCPU=%f,\t\t rUDT=%f\n", i, rCpu, rUdt);
            rslt=false;
        }
    }

    std::cout<<std::endl;

    if(rslt){
        std::cout<<"Test \""<<testName<<"\" is successfully verified."<<std::endl;
    }

    return (rslt)? 0 : 1;
}

int main(int argc, char **argv) {
    int rslt0 = TestReduceSum4D<16>("ReduceSum4D_TTTF", {2,2,2,17}, 1);
    rslt0 += TestReduceSum4D<16>("ReduceSum4D_TTTF", {2,2,2,64}, 1);
    rslt0 += TestReduceSum4D<16>("ReduceSum4D_TTTF", {2,2,2,119}, 1);
    return rslt0;
}

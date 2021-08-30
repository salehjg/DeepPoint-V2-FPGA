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
using namespace ConfigTaskReluSqrtSquare;

extern "C"
void task_relu_sqrt_square(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned len,
        const unsigned mode);

void GoldReluSqrtSquare(
    const CONFIG_DTYPE *inputTn,
    CONFIG_DTYPE *outputTn,
    const unsigned lenItems,
    const unsigned mode){

    for(unsigned iter=0; iter<lenItems; iter++){
        const CONFIG_DTYPE val = inputTn[iter];
        if(mode==ConfigTaskReluSqrtSquare::ModeRelu){
            outputTn[iter] = (val>0)?val:0;
        }else if(mode==ConfigTaskReluSqrtSquare::ModeSqrt){
            outputTn[iter] = sqrt(val);
        }else if(mode==ConfigTaskReluSqrtSquare::ModeSquare){
            outputTn[iter] = (CONFIG_DTYPE)(val*val);
        } 
    }
}

template<unsigned vecSize>
int TestReluSqrtSquare(
    const string& testName,
    const std::vector<unsigned> shape,
    const bool runRelu,
    const bool runSqrt,
    const bool runSquare){

    assert(
        (runRelu&& !runSqrt&& !runSquare)||
        (!runRelu&& runSqrt&& !runSquare)||
        (!runRelu&& !runSqrt&& runSquare)
        );

    const unsigned mode = runRelu?ConfigTaskReluSqrtSquare::ModeRelu:
                          runSqrt?ConfigTaskReluSqrtSquare::ModeSqrt:
                          runSquare?ConfigTaskReluSqrtSquare::ModeSquare:
                          100;

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;
    const unsigned rank = shape.size();
    const unsigned lastDim = shape[rank-1]; 
    const unsigned lastDimPadded = MakeDivisible<unsigned>(lastDim, CONFIG_M_AXI_WIDTH);

    unsigned lenInput = 1;
    unsigned lenInputPadded = 1;
    for(unsigned i=0; i<rank; i++){
        lenInput *= shape[i];
        lenInputPadded *= (i!=rank-1)? shape[i]: lastDimPadded;
    }

    const unsigned lenOutput = lenInput;
    const unsigned lenOutputUdt = lenInputPadded;

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

    PadTensor<CONFIG_DTYPE>(hostInputTn, hostInputTnPadded, lenInput/lastDim, lastDim, lastDimPadded);

    auto deviceInputTn = Pack<vecSize, CONFIG_DTYPE>(hostInputTnPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostUDT);

    task_relu_sqrt_square(
        deviceInputTn.data(),
        deviceOutputTn.data(),
        lenInputPadded/vecSize,
        mode);

    GoldReluSqrtSquare(
        hostInputTn.data(), 
        hostGold.data(), 
        lenInput,
        mode);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    // UnpadTensor is not needed as this kernel
    UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUDT, lenOutput/lastDim, lastDimPadded, lastDim);
    bool rslt = true;

    for(unsigned iter=0; iter<lenOutput; iter++){ 
        unsigned indxCpu = iter;
        unsigned indxUdt = iter;
        CONFIG_DTYPE rCpu = hostGold[indxCpu];
        CONFIG_DTYPE rUdt = hostUDT[indxUdt];
        CONFIG_DTYPE diff = rCpu - rUdt;
        if(abs(diff)>1e-02){
            printf("iter= (%03d)\trCPU=%f,\t\t rUDT=%f\n", iter, rCpu, rUdt);
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
    int rslt = 0;
    rslt += TestReluSqrtSquare<16>("ReluSqrtSquare: ReLU", {2,2,17}, true, false, false);
    rslt += TestReluSqrtSquare<16>("ReluSqrtSquare: Sqrt", {2,2,64}, false, true, false);
    rslt += TestReluSqrtSquare<16>("ReluSqrtSquare: Square", {2,1024,50}, false, false, true);
    return rslt;
}

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
using namespace ConfigTaskTopK;

extern "C"
void task_gather(
    const MemoryPackF_t *inputTn,
    const unsigned *indicesTn,
    MemoryPackF_t *outputTn,
    unsigned indicesAxis,
    unsigned inputDim0,
    unsigned inputDim1,
    unsigned inputDim2,
    unsigned indicesDim0,
    unsigned indicesDim1,
    unsigned indicesDim2);

void CpuGoldGather(
    const CONFIG_DTYPE* buffInput,
    const unsigned* buffIndices,
    CONFIG_DTYPE* buffOutput,
    const unsigned axis,
    const unsigned sizeB,
    const unsigned sizeN,
    const unsigned sizeK,
    const unsigned sizeD){
    unsigned indxS1, indxS2, indxD;
    for(unsigned b=0;b<sizeB;b++){
        for(unsigned n=0;n<sizeN;n++){
            for(unsigned k=0;k<sizeK;k++){
                indxS1 = b*sizeN*sizeK + n*sizeK + k;
                for(unsigned d=0;d<sizeD;d++)
                {
                    indxD = b*sizeN*sizeK*sizeD + n*sizeK*sizeD + k*sizeD + d;
                    indxS2 = b*sizeN*sizeD +
                             buffIndices[indxS1]*sizeD +
                             d;
                    buffOutput[indxD] = buffInput[indxS2];
                }
            }
        }
    }
}

void InitIndices(
    unsigned* buffIndices,
    const unsigned B,
    const unsigned N,
    const unsigned K){

    assert(K<N);

    for(unsigned b=0; b<B; b++){
        for(unsigned n=0; n<N; n++){
            for(unsigned k=0; k<K; k++){
                const unsigned indx = b*N*K+ n*K+ k;
                buffIndices[indx] = N-K;
            }
        }
    }
}

template<unsigned vecSize>
int TestGather(
    const string& testName,
    const unsigned axis,
    const unsigned sizeB,
    const unsigned sizeN,
    const unsigned sizeK,
    const unsigned sizeD){

    assert(axis==1);
    assert(sizeK<sizeN);

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl; 

    const unsigned lenInput = sizeB*sizeN*sizeD;
    const unsigned lenIndices = sizeB*sizeN*sizeK;
    const unsigned lenOutput = sizeB*sizeN*sizeK*sizeD;

    const unsigned sizeDPadded = MakeDivisible<unsigned>(sizeD, CONFIG_M_AXI_WIDTH); 
    const unsigned sizeKPadded = MakeDivisible<unsigned>(sizeK, CONFIG_M_AXI_WIDTH); 

    const unsigned lenInputPadded = sizeB*sizeN*sizeDPadded;
    const unsigned lenIndicesPadded = sizeB*sizeN*sizeKPadded;
    const unsigned lenOutputPadded = sizeB*sizeN*sizeK*sizeDPadded;

    std::vector<CONFIG_DTYPE> hostInputTn(lenInput); 
    std::vector<unsigned> hostIndicesTn(lenIndices);
    std::vector<CONFIG_DTYPE> hostGold(lenOutput);

    std::vector<CONFIG_DTYPE> hostInputTnPadded(lenInputPadded);
    std::vector<unsigned> hostIndicesTnPadded(lenIndicesPadded);
    std::vector<CONFIG_DTYPE> hostUDT(lenOutputPadded);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTn.begin(), hostInputTn.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    InitIndices(hostIndicesTn.data(), sizeB, sizeN, sizeK);

    PadTensor<CONFIG_DTYPE>(hostInputTn, hostInputTnPadded, sizeB*sizeN, sizeD, sizeDPadded);
    PadTensor<unsigned >(hostIndicesTn, hostIndicesTnPadded, sizeB*sizeN, sizeK, sizeKPadded);

    auto deviceInputTn = Pack<vecSize, CONFIG_DTYPE>(hostInputTnPadded);
    //auto deviceIndicesTn = Pack<vecSize, unsigned>(hostIndicesTnPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostUDT);

    //task_topk(deviceInputTn.data(), deviceOutputTn.data(), dim0, dim1, kValue, vecsPerSlice, vecsPerOutputSlice);
    task_gather(
        deviceInputTn.data(),
        hostIndicesTnPadded.data(),
        deviceOutputTn.data(), 
        axis,
        sizeB,sizeN,sizeD,
        sizeB,sizeN,sizeK);
    CpuGoldGather(
        hostInputTn.data(), 
        hostIndicesTn.data(), 
        hostGold.data(), 
        axis, 
        sizeB, sizeN, sizeK, sizeD);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    bool rslt = true;

    for(unsigned b=0;b<sizeB;b++){
        for(unsigned n=0;n<sizeN;n++){
            for(unsigned k=0;k<sizeK;k++){
                for(unsigned d=0;d<sizeD;d++){
                    const unsigned indxGold = b*sizeN*sizeK*sizeD + n*sizeK*sizeD + k*sizeD + d;
                    const unsigned indxUdt = indxGold;
                    CONFIG_DTYPE rCpu = hostGold[indxGold];
                    CONFIG_DTYPE rUdt = hostOutputTn[indxUdt];
                    CONFIG_DTYPE diff = rCpu - rUdt;
                    if(abs(diff) > 1e-2){
                        printf("Index(B,N,K,D)= (%03d, %03d, %03d, %03d)\t\trCPU=%f, rUDT=%f\n",
                                b, n, k, d,
                                rCpu,rUdt);
                        rslt=false;
                    }
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
    rslt += TestGather<16>("GatherAxis1", 1, 5, 1024, 20, 64);
    return rslt;
}

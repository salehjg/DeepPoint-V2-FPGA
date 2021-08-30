#include "PaddingCpu.h"
#include "Utility.h"
#include "AxiHelper.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <string>
#include <cassert>

using namespace std;

extern "C"
void task_pad_unpad(
    const MemoryPackF_t *inputTn,
    MemoryPackF_t *outputTn,
    const unsigned mode,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned pad_dim1Padded,
    const unsigned pad_lcm,
    const unsigned unpad_dim1Unpadded);

template<unsigned int vecSize>
int TestPadding(
    const string testName, 
    const unsigned int dim0, 
    const unsigned int dim1, 
    const unsigned int dim1Padded){

    assert(dim1<dim1Padded);
    assert(dim1Padded%vecSize==0);

    bool isSubVec = (dim1<vecSize);
    unsigned int lenInput  = MakeDivisible<unsigned int>(dim0*dim1, vecSize);
    unsigned int lenOutput = MakeDivisible<unsigned int>(dim0*dim1Padded, vecSize);

    const unsigned int gcd = __gcd(dim1, vecSize);
    const unsigned int lcm = (dim1*vecSize)/(gcd);

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;
    cout<<"InputShape: "<<dim0<<"x"<<dim1<<endl;
    cout<<"Padding Dimension: "<<"1"<<endl;
    cout<<"The Dimension Before Padding: "<<dim1<<endl;
    cout<<"The Dimension After Padding: "<<dim1Padded<<endl;
    cout<<"Vector Size: "<<vecSize<<endl;
    if(isSubVec){
        cout<<"GCD(dim1,vecSize): "<<gcd<<endl;
        cout<<"LCM(dim1,vecSize): "<<lcm<<endl;
    }

    std::vector<CONFIG_DTYPE> hostInputTn(lenInput); 
    std::vector<CONFIG_DTYPE> hostGold(lenOutput);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTn.begin(), hostInputTn.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    const auto deviceInputTn = Pack<vecSize, float>(hostInputTn);
    auto deviceOutputTn = Pack<vecSize, float>(hostGold);

    task_pad_unpad(deviceInputTn.data(), deviceOutputTn.data(), 1, dim0, dim1, dim1Padded, lcm, 0);
    PadTensor<CONFIG_DTYPE>(hostInputTn, hostGold, dim0, dim1, dim1Padded);

    const auto hostOutputTn = Unpack<vecSize, float>(deviceOutputTn);
    bool rslt = true;

    for(int d0=0; d0<dim0; d0++){
        for(int d1=0; d1<dim1; d1++){ // this loop should be over dim1 instead of dim1Padded.
            unsigned int indx = d0*dim1Padded+d1;
            CONFIG_DTYPE diff = (hostOutputTn[indx] - hostGold[indx]);
            if(abs(diff)>1e-02){
                std::cout<<"Mismatch at d0: "<< d0 <<", d1: "<< d1 << std::endl;
                std::cout<<"Value: "<< hostOutputTn[indx] << std::endl;
                std::cout<<"Gold: "<< hostGold[indx] << std::endl;
                rslt = false;
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

    //Sub-vec padding is disabled in the kernel, as Cpu last dim padding is enabled.
    //int rsltSubVec = TestPadding<16>("SubVecPadding", 32, 6, 16);

    int rsltSuperVec = TestPadding<16>("SuperVecPadding", 32, 64, 128);
    return /*rsltSubVec+*/rsltSuperVec;
}

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
using namespace ConfigTaskConcat;

extern "C"
void task_concat(
    const MemoryPackF_t *inputTn1,
    const MemoryPackF_t *inputTn2,
    MemoryPackF_t *outputTn,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dimA3,
    const unsigned dimB3,
    const int concatDim);

void GoldConcat(
    const CONFIG_DTYPE* buffA,
    const CONFIG_DTYPE* buffB,
    CONFIG_DTYPE* buffR,
    const unsigned dim0,
    const unsigned dim1,
    const unsigned dim2,
    const unsigned dim3A,
    const unsigned dim3B,
    const unsigned dim3R){

    for(unsigned d0=0; d0<dim0; d0++){ 
        for(unsigned d1=0; d1<dim1; d1++){ 
            for(unsigned d2=0; d2<dim2; d2++){ 
                for(unsigned d3=0; d3<dim3A; d3++){ 
                    const unsigned indxD = d0*dim1*dim2*dim3R + 
                                           d1*dim2*dim3R + 
                                           d2*dim3R + 
                                           d3;
                    const unsigned indxS = d0*dim1*dim2*dim3A + 
                                           d1*dim2*dim3A + 
                                           d2*dim3A + 
                                           d3;
                    buffR[indxD] = buffA[indxS];
                }

                for(unsigned d3=0; d3<dim3B; d3++){ 
                    const unsigned indxD = d0*dim1*dim2*dim3R + 
                                           d1*dim2*dim3R + 
                                           d2*dim3R +
                                           (dim3A+d3);
                    const unsigned indxS = d0*dim1*dim2*dim3B + 
                                           d1*dim2*dim3B + 
                                           d2*dim3B + 
                                           (d3);
                    buffR[indxD] = buffB[indxS];
                }
            }
        }
    }
}

template<unsigned vecSize>
int TestConcat(
    const string& testName,
    const std::vector<unsigned> shapeA,
    const std::vector<unsigned> shapeB,
    const int axis){

    const unsigned rank = shapeA.size();
    const unsigned rankB = shapeB.size();
    assert(rank == 4);
    assert(rank == rankB);
    assert(shapeA[0]==shapeB[0]);
    assert(shapeA[1]==shapeB[1]);
    assert(shapeA[2]==shapeB[2]);

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;

    unsigned dim0=0, dim1=0, dim2=0, dim3A=0, dim3B=0, dim3R=0;
    unsigned dim3APadded=0, dim3BPadded=0, dim3RPadded=0;
    unsigned lenInputA=0, lenInputAPadded=0, 
             lenInputB=0, lenInputBPadded=0, 
             lenOutput=0, lenOutputUdt=0;

    dim0=shapeA[0];
    dim1=shapeA[1]; 
    dim2=shapeA[2]; 
    dim3A=shapeA[3]; 
    dim3B=shapeB[3];
    dim3R=dim3A+dim3B; 

    dim3APadded=MakeDivisible<unsigned>(dim3A, CONFIG_M_AXI_WIDTH);
    dim3BPadded=MakeDivisible<unsigned>(dim3B, CONFIG_M_AXI_WIDTH);
    dim3RPadded=MakeDivisible<unsigned>(dim3R, CONFIG_M_AXI_WIDTH);

    lenInputA=dim0*dim1*dim2*dim3A;
    lenInputB=dim0*dim1*dim2*dim3B; 
    lenOutput=dim0*dim1*dim2*dim3R;

    lenInputAPadded=dim0*dim1*dim2*dim3APadded;
    lenInputBPadded=dim0*dim1*dim2*dim3BPadded; 
    lenOutputUdt=dim0*dim1*dim2*dim3RPadded;
    
    std::vector<CONFIG_DTYPE> hostInputTnA(lenInputA); 
    std::vector<CONFIG_DTYPE> hostInputTnB(lenInputB); 
    std::vector<CONFIG_DTYPE> hostInputTnAPadded(lenInputAPadded);
    std::vector<CONFIG_DTYPE> hostInputTnBPadded(lenInputBPadded);
    std::vector<CONFIG_DTYPE> hostGold(lenOutput);
    std::vector<CONFIG_DTYPE> hostUDT(lenOutputUdt);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTnA.begin(), hostInputTnA.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });
    std::for_each(hostInputTnB.begin(), hostInputTnB.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    PadTensor<CONFIG_DTYPE>(hostInputTnA, hostInputTnAPadded, dim0*dim1*dim2, dim3A, dim3APadded);
    PadTensor<CONFIG_DTYPE>(hostInputTnB, hostInputTnBPadded, dim0*dim1*dim2, dim3B, dim3BPadded);
    
    auto deviceInputTnA = Pack<vecSize, CONFIG_DTYPE>(hostInputTnAPadded);
    auto deviceInputTnB = Pack<vecSize, CONFIG_DTYPE>(hostInputTnBPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostUDT);

    task_concat(
        deviceInputTnA.data(),
        deviceInputTnB.data(),
        deviceOutputTn.data(),
        dim0,
        dim1,
        dim2,
        dim3A,
        dim3B,
        axis);

    GoldConcat(
        hostInputTnA.data(),
        hostInputTnB.data(),
        hostGold.data(),
        dim0,
        dim1,
        dim2,
        dim3A,
        dim3B,
        dim3R);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUDT, dim0*dim1*dim2, dim3RPadded, dim3R);

    bool rslt = true;

    for(unsigned d0=0; d0<dim0; d0++){ 
        for(unsigned d1=0; d1<dim1; d1++){ 
            for(unsigned d2=0; d2<dim2; d2++){ 
                for(unsigned d3=0; d3<dim3R; d3++){ 
                    const unsigned indxCpu = d0*dim1*dim2*dim3R + 
                                             d1*dim2*dim3R + 
                                             d2*dim3R + 
                                             d3;
                    const unsigned indxUdt = indxCpu;
                    CONFIG_DTYPE rCpu = hostGold[indxCpu];
                    CONFIG_DTYPE rUdt = hostUDT[indxUdt];
                    CONFIG_DTYPE diff = rCpu - rUdt;
                    if(abs(diff)>1e-02){
                        printf("d0= (%03d)\td1= (%03d)\td2= (%03d)\td3= (%03d)\trCPU=%f,\t\t rUDT=%f\n", 
                            d0, d1, d2, d3, rCpu, rUdt);
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
    rslt += TestConcat<16>("Concat Sub-vec", {2,2,2,3}, {2,2,2,3}, 3);
    rslt += TestConcat<16>("Concat Super-vec", {2,2,2,192}, {2,2,2,128}, 3);
    return rslt;
}

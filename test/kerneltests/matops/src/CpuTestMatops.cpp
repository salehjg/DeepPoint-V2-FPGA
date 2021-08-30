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
#include "GoldMatops.h"

using namespace std;
using namespace ConfigTaskMatOps;

extern "C"
void task_matops(
        const MemoryPackF_t *inputTn1,
        const MemoryPackF_t *inputTn2,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned dim3,
        const unsigned dim0B,
        const unsigned dim1B,
        const unsigned dim2B,
        const unsigned dim3B, 
        const int rankA,
        const int rankB,
        const int mode);

template<unsigned int vecSize>
int TestMatops(
    const string testName, 
    const std::vector<unsigned> &shapeA,
    const std::vector<unsigned> &shapeB,
    int mode){

    const unsigned rankA = shapeA.size();
    const unsigned rankB = shapeB.size();
    const bool isConstantB = (rankB==1 && shapeB[0]==1);

    /*if(rankA==4 && rankB==1 && isConstantB){//DBG ONLY
        int a;
        a=10;
    }*/

    const unsigned lastDimPaddedA = MakeDivisible<unsigned>(shapeA[rankA-1], CONFIG_M_AXI_WIDTH);
    const unsigned lastDimPaddedB = MakeDivisible<unsigned>(shapeB[rankB-1], CONFIG_M_AXI_WIDTH);

    unsigned lenInputA = 1; for(unsigned i=0; i<rankA; i++){lenInputA*=shapeA[i];}
    unsigned lenInputB = 1; for(unsigned i=0; i<rankB; i++){lenInputB*=shapeB[i];}
    unsigned lenOutput = lenInputA;

    unsigned lenInputAPadded = lastDimPaddedA; for(unsigned i=0; i<rankA-1; i++){lenInputAPadded*=shapeA[i];}
    unsigned lenInputBPadded = lastDimPaddedB; for(unsigned i=0; i<rankB-1; i++){lenInputBPadded*=shapeB[i];}
    unsigned lenOutputPadded = lenInputAPadded;

    unsigned bSizeA = 1; for(unsigned i=0; i<rankA-1; i++){bSizeA*=shapeA[i];}
    unsigned bSizeB = 1; for(unsigned i=0; i<rankB-1; i++){bSizeB*=shapeB[i];}

    std::vector<CONFIG_DTYPE> hostInputTnA(lenInputA); 
    std::vector<CONFIG_DTYPE> hostInputTnB(lenInputB); 
    std::vector<CONFIG_DTYPE> hostGold(lenOutput);
    std::vector<CONFIG_DTYPE> hostUdtUnpadded(lenOutput);

    std::vector<CONFIG_DTYPE> hostInputTnAPadded(lenInputAPadded); 
    std::vector<CONFIG_DTYPE> hostInputTnBPadded(lenInputBPadded); 
    std::vector<CONFIG_DTYPE> hostGoldPadded(lenOutputPadded);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    std::for_each(hostInputTnA.begin(), hostInputTnA.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });
    std::for_each(hostInputTnB.begin(), hostInputTnB.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    PadTensor<CONFIG_DTYPE>(hostInputTnA, hostInputTnAPadded, bSizeA, shapeA[rankA-1], lastDimPaddedA);
    PadTensor<CONFIG_DTYPE>(hostInputTnB, hostInputTnBPadded, bSizeB, shapeB[rankB-1], lastDimPaddedB);

    const auto deviceInputTnA = Pack<vecSize, CONFIG_DTYPE>(hostInputTnAPadded);
    const auto deviceInputTnB = Pack<vecSize, CONFIG_DTYPE>(hostInputTnBPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostGoldPadded);

    unsigned _dim0,_dim1,_dim2,_dim3;
    unsigned _dim0B,_dim1B,_dim2B,_dim3B;
    switch(rankA){
        case 4:{
            _dim0=shapeA[0];
            _dim1=shapeA[1];
            _dim2=shapeA[2];
            _dim3=shapeA[3];
            break;
        }
        case 3:{
            _dim0=1;
            _dim1=shapeA[0];
            _dim2=shapeA[1];
            _dim3=shapeA[2];
            break;
        }
        case 2:{
            _dim0=1;
            _dim1=1;
            _dim2=shapeA[0];
            _dim3=shapeA[1];
            break;
        }
        case 1:{
            _dim0=1;
            _dim1=1;
            _dim2=1;
            _dim3=shapeA[0];
            break;
        }
        default:{
            _dim0=-1;
            _dim1=-1;
            _dim2=-1;
            _dim3=-1;
        }
    }

    switch(rankB){
        case 4:{
            _dim0B=shapeB[0];
            _dim1B=shapeB[1];
            _dim2B=shapeB[2];
            _dim3B=shapeB[3];
            break;
        }
        case 3:{
            _dim0B=0;
            _dim1B=shapeB[0];
            _dim2B=shapeB[1];
            _dim3B=shapeB[2];
            break;
        }
        case 2:{
            _dim0B=0;
            _dim1B=0;
            _dim2B=shapeB[0];
            _dim3B=shapeB[1];
            break;
        }
        case 1:{
            _dim0B=0;
            _dim1B=0;
            _dim2B=0;
            _dim3B=shapeB[0];
            break;
        }
        default:{
            _dim0B=-1;
            _dim1B=-1;
            _dim2B=-1;
            _dim3B=-1;
        }
    }

    //if(isConstantB){std::printf("***cte:%f\n",deviceInputTnB[0][0]);}//DBG ONLY
    task_matops(
            deviceInputTnA.data(),
            deviceInputTnB.data(),
            deviceOutputTn.data(),
            _dim0,
            _dim1,
            _dim2,
            _dim3,
            _dim0B,
            _dim1B,
            _dim2B,
            _dim3B,
            rankA,
            rankB,
            mode);
    GoldMatops<CONFIG_DTYPE>(
            hostInputTnA.data(),
            hostInputTnB.data(),
            hostGold.data(),
            _dim0,
            _dim1,
            _dim2,
            _dim3,
            _dim0B,
            _dim1B,
            _dim2B,
            _dim3B,
            rankA,
            rankB,
            mode);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUdtUnpadded, bSizeA, lastDimPaddedA, _dim3);
    
    bool rslt = true;

    for(unsigned d0=0; d0<_dim0 && rslt; d0++){
        for(unsigned d1=0; d1<_dim1 && rslt; d1++){
            for(unsigned d2=0; d2<_dim2 && rslt; d2++){
                for(unsigned d3=0; d3<_dim3 && rslt; d3++){
                    const unsigned indx = d0*_dim1*_dim2*_dim3+
                                        d1*_dim2*_dim3+
                                        d2*_dim3+
                                        d3;
                    CONFIG_DTYPE rCpu = hostGold[indx];
                    CONFIG_DTYPE rUdt = hostUdtUnpadded[indx];
                    CONFIG_DTYPE diff = (rUdt - rCpu); 
                    if(abs(diff)>1e-02){
                        std::printf("Mismatch at [d0,d1,d2,d3]=[%d,%d,%d,%d] Gold=%f, Udt=%f\n",
                            d0,d1,d2,d3,
                            rCpu,
                            rUdt);
                        rslt = false;
                    }
                }
            }
        }
    }

    if(rslt){
        std::cout<<"Test \""<<testName<<"\", Mode="<<mode<<" is successfully verified."<<std::endl;
    }else{
        std::cout<<"Test \""<<testName<<"\", Mode="<<mode<<" is failed."<<std::endl;
    }

    return (rslt)? 0 : 1;
}

int RunTests(unsigned _dim0, unsigned _dim1, unsigned _dim2, unsigned _dim3){
    int result=0;
    unsigned dim0, dim1, dim2, dim3;
    dim0 = _dim0;
    dim1 = _dim1;
    dim2 = _dim2;
    dim3 = _dim3;

    {
        std::printf("\nRunning tests for 4D,nD n<=4 [%d,%d,%d,%d]...\n",dim0,dim1,dim2,dim3);
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:4D,4D", {dim0, dim1, dim2, dim3}, {dim0, dim1, dim2, dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:4D,3D", {dim0, dim1, dim2, dim3}, {dim1, dim2, dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:4D,2D", {dim0, dim1, dim2, dim3}, {dim2, dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:4D,1D", {dim0, dim1, dim2, dim3}, {dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:4D,cte", {dim0, dim1, dim2, dim3}, {1}, mode);
        }
    }
    {
        dim0=-1;
        std::printf("\nRunning tests for 3D,nD n<=3 [%d,%d,%d]...\n",dim1,dim2,dim3);
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:3D,3D", {dim1, dim2, dim3}, {dim1, dim2, dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:3D,2D", {dim1, dim2, dim3}, {dim2, dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:3D,1D", {dim1, dim2, dim3}, {dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:3D,cte", {dim1, dim2, dim3}, {1}, mode);
        }
    }
    {
        dim0=-1; dim1=-1;
        std::printf("\nRunning tests for 2D,nD n<=2 [%d,%d]...\n",dim2,dim3);
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:2D,2D", {dim2, dim3}, {dim2, dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:2D,1D", {dim2, dim3}, {dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:2D,cte", {dim2, dim3}, {1}, mode);
        }
    }
    {
        dim0=-1; dim1=-1; dim2=-1;
        std::printf("\nRunning tests for 1D,nD n<=1 [%d]...\n",dim3);
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:1D,1D", {dim3}, {dim3}, mode);
        }
        for(int mode=0; mode<4; mode++){
            result += TestMatops<16>("MatOps:1D,cte", {dim3}, {1}, mode);
        }
    }

    return result;
}

int main(int argc, char **argv) {
    int result = 0;

    result += RunTests(2,5,32,32);
    result += RunTests(2,5,32,6);
    result += RunTests(2,5,32,17);
    result += RunTests(2,5,6,6);

    if(result==0){
        cout<<"\n========\nAll of the tests are run successfully."<<endl;
    }else{
        cout<<"\n========\nAll or some of the tests are failed."<<endl;
    }
    return result;
}

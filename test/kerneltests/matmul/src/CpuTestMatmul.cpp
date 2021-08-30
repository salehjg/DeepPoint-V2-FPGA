#include "PaddingCpu.h"
#include "Utility.h"
#include "AxiHelper.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <string>
#include <cassert>

using namespace std;
using namespace ConfigTaskMatMul;

extern "C"
void task_matmul(
        const CONFIG_DTYPE *inputTn1,
        const MemoryPackF_t *inputTn2,
        MemoryPackF_t *outputTn,
        const unsigned sizeBatch,
        const unsigned sizeN,
        const unsigned sizeK,
        const unsigned sizeM);

void GoldMatmul(
    const CONFIG_DTYPE* matA,
    const CONFIG_DTYPE* matB,
    CONFIG_DTYPE* matC,
    const unsigned sizeBatch,
    const unsigned sizeN,
    const unsigned sizeK,
    const unsigned sizeM){
    unsigned indxS1, indxS2, indxD;
    for(unsigned batch=0; batch<sizeBatch; batch++){
        for(unsigned n=0; n<sizeN; n++){
            for(unsigned m=0; m<sizeM; m++){
                CONFIG_DTYPE acc = 0;
                for(unsigned k=0; k<sizeK; k++){
                    indxS1 = batch*sizeN*sizeK + n*sizeK + k;
                    indxS2 = batch*sizeK*sizeM + k*sizeM + m;
                    acc += matA[indxS1] * matB[indxS2];
                }
                indxD = batch*sizeN*sizeM + n*sizeM + m;
                matC[indxD] = acc;
            }
        }
    }
}

void InitTensor(
        CONFIG_DTYPE *tn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2,
        const unsigned mode){
    for(unsigned b=0; b<dim0; b++){
        for(unsigned d1=0; d1<dim1; d1++){
            for(unsigned d2=0; d2<dim2; d2++){
                const unsigned indxS = b*dim1*dim2 + d1*dim2 + d2;
                if(mode==0){
                    CONFIG_DTYPE val = (float)b*10.f + (float)d1 + (float)d2/100.f;
                    tn[indxS] = val;
                }else if(mode==1){
                    CONFIG_DTYPE val = 1.0f;
                    tn[indxS] = val;
                }else if(mode==2){
                    CONFIG_DTYPE val = 2.0f;
                    tn[indxS] = val;
                }

            }
        }
    }
}

template<unsigned int vecSize>
int TestMatmul(
    const string testName, 
    const std::vector<unsigned> &shapeA,
    const std::vector<unsigned> &shapeB){

    // MatA's  shape = [dim0, dim1, dim2] = [batchSize, sizeN, sizeK] = [Batch, Height, Width]; Row-major
    // MatB's  shape = [dim0, dim1, dim2] = [batchSize, sizeK, sizeM] = [Batch, Height, Width]; Row-major
    // MatC=AB shape = [dim0, dim1, dim2] = [batchSize, sizeN, sizeM] = [Batch, Height, Width]; Row-major

    const unsigned rankA = shapeA.size();
    const unsigned rankB = shapeB.size(); 

    assert(rankA==rankB);
    assert(shapeA[0]==shapeB[0]); //batchSize
    assert(shapeA[2]==shapeB[1]); // K 

    const unsigned sizeBatch = shapeA[0];
    const unsigned sizeN = shapeA[1];
    const unsigned sizeK = shapeA[2];
    const unsigned sizeM = shapeB[2];

    const unsigned lastDimPaddedA = MakeDivisible<unsigned>(sizeK, CONFIG_M_AXI_WIDTH);
    const unsigned lastDimPaddedB = MakeDivisible<unsigned>(sizeM, CONFIG_M_AXI_WIDTH);
    const unsigned lastDimPaddedC = MakeDivisible<unsigned>(sizeM, CONFIG_M_AXI_WIDTH);

    unsigned lenInputA = sizeBatch*sizeN*sizeK;
    unsigned lenInputB = sizeBatch*sizeK*sizeM;
    unsigned lenOutput = sizeBatch*sizeN*sizeM;

    unsigned lenInputAPadded = sizeBatch*sizeN*lastDimPaddedA;
    unsigned lenInputBPadded = sizeBatch*sizeK*lastDimPaddedB;
    unsigned lenOutputPadded = sizeBatch*sizeN*lastDimPaddedC;

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

    //InitTensor(hostInputTnA.data(), sizeBatch, sizeN, sizeK,0);
    //InitTensor(hostInputTnB.data(), sizeBatch, sizeK, sizeM,0);

    PadTensor<CONFIG_DTYPE>(hostInputTnA, hostInputTnAPadded, sizeBatch*sizeN, sizeK, lastDimPaddedA);
    PadTensor<CONFIG_DTYPE>(hostInputTnB, hostInputTnBPadded, sizeBatch*sizeK, sizeM, lastDimPaddedB);

    const auto deviceInputTnB = Pack<vecSize, CONFIG_DTYPE>(hostInputTnBPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostGoldPadded);

    GoldMatmul(
            hostInputTnA.data(),
            hostInputTnB.data(),
            hostGold.data(),
            sizeBatch,
            sizeN,
            sizeK,
            sizeM);

    task_matmul(
            hostInputTnAPadded.data(),
            deviceInputTnB.data(),
            deviceOutputTn.data(),
            sizeBatch,
            sizeN,
            sizeK,
            sizeM);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUdtUnpadded, sizeBatch*sizeN, lastDimPaddedC, sizeM);
    
    bool rslt = true;

    for(unsigned batch=0; batch<sizeBatch; batch++){
        for(unsigned n=0; n<sizeN; n++){
            for(unsigned m=0; m<sizeM; m++){
                
                const unsigned indx = batch*sizeN*sizeM+
                                      n*sizeM+
                                      m;
                CONFIG_DTYPE rCpu = hostGold[indx];
                CONFIG_DTYPE rUdt = hostUdtUnpadded[indx];
                CONFIG_DTYPE diff = (rUdt - rCpu); 
                if(abs(diff)>1e-02){
                    std::printf("Mismatch at [batch,n,m]=[%d,%d,%d] Gold=%f, Udt=%f\n",
                        batch,n,m,
                        rCpu,
                        rUdt);
                    rslt = false;
                }

            }
        }
    }

    if(rslt){
        std::cout<<"Test \""<<testName<<"\" is successfully verified."<<std::endl;
    }else{
        std::cout<<"Test \""<<testName<<"\" is failed."<<std::endl;
    }

    return (rslt)? 0 : 1;
}

int main(int argc, char **argv) {
    int result = 0;

    {
        unsigned B=1, N=5, K=2, M=5;
        result += TestMatmul<16>("Matmul:3D,3D", {B, N, K}, {B, K, M});
    }
    {
        unsigned B=1, N=5, K=1024, M=512;
        result += TestMatmul<16>("Matmul:3D,3D", {B, N, K}, {B, K, M});
    }
    {
        unsigned B=5, N=3, K=3, M=3;
        result += TestMatmul<16>("Matmul:3D,3D", {B, N, K}, {B, K, M});
    }
    {
        unsigned B=5, N=1024, K=3, M=3;
        result += TestMatmul<16>("Matmul:3D,3D", {B, N, K}, {B, K, M});
    }
    {
        unsigned B=1, N=4, K=17, M=16;
        result += TestMatmul<16>("Matmul:3D,3D", {B, N, K}, {B, K, M});
    }
    if(result==0){
        cout<<"\n========\nAll of the tests are run successfully."<<endl;
    }else{
        cout<<"\n========\nAll or some of the tests are failed."<<endl;
    }
    return result;
}

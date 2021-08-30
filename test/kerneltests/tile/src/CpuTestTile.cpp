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
using namespace ConfigTaskTile;

extern "C"
void task_tile(
        const MemoryPackF_t *inputTn,
        MemoryPackF_t *outputTn,
        const unsigned dim0,
        const unsigned dim1,
        const unsigned dim2, 
        const unsigned rank,
        const unsigned tileAxis,
        const unsigned tileSize);

void GoldTile(
    const CONFIG_DTYPE *inputTn,
    CONFIG_DTYPE *outputTn,
    const std::vector<unsigned> &shape,
    const int tileAxis,
    const int tileSize){

    const unsigned rank = shape.size();
    assert(rank<=3 && rank>=2);
    assert(
        (rank==2 && tileAxis==1)||
        (rank==2 && tileAxis==2)||
        (rank==3 && tileAxis==2)
        );

    unsigned _dim0=0, _dim1=0, _dim2=0;
    if(rank==2 && tileAxis==1){
        //input: BxN ===> output: BxTxN ===> lastDim: N,(dim1)
        _dim0 = shape[0];
        _dim1 = 1;
        _dim2 = shape[1];
    }else if(rank==2 && tileAxis==2){
        //input: BxN ===> output: BxNxT ===> lastDim: T,(tileSize)
        _dim0 = shape[0];
        _dim1 = shape[1];
        _dim2 = 1;
    }else if(rank==3 && tileAxis==2){
        //input: BxNxD ===> output: BxNxTxD ===> lastDim: D,(dim2)
        _dim0 = shape[0]*shape[1];
        _dim1 = 1;
        _dim2 = shape[2];
    }

    if(rank==2 && tileAxis==2){
        for(unsigned d0=0; d0<_dim0; d0++){
            for(unsigned d1=0; d1<_dim1; d1++){
                const unsigned indxS = d0*_dim1 + d1;
                CONFIG_DTYPE val = inputTn[indxS];
                for(unsigned t=0; t<tileSize; t++){
                    const unsigned indxD = d0*_dim1*tileSize + d1*tileSize + t;
                    outputTn[indxD] = val;
                }
            }       
        }
    }else if((rank==2 && tileAxis==1) || (rank==3 && tileAxis==2)){
        for(unsigned d0=0; d0<_dim0; d0++){
            for(unsigned d2=0; d2<_dim2; d2++){
                const unsigned indxS = d0*_dim2 + d2;
                CONFIG_DTYPE val = inputTn[indxS];
                for(unsigned t=0; t<tileSize; t++){
                    const unsigned indxD = d0*tileSize*_dim2 + t*_dim2 + d2;
                    outputTn[indxD] = val;
                }
            }       
        }
    }else{
        //NYI
        assert(false);
    }

}

template<unsigned vecSize>
int TestTile(
    const string& testName,
    const std::vector<unsigned> shape,
    const int tileAxis,
    const int tileSize){

    const unsigned rank = shape.size();
    assert(rank<=3 && rank>=2);
    assert(
        (rank==2 && tileAxis==1)||
        (rank==2 && tileAxis==2)||
        (rank==3 && tileAxis==2)
        );

    cout<<"=================================================="<<endl;
    cout<<"TestName: "<< testName <<endl;

    unsigned dim0=0, dim1=0, dim2=0;
    unsigned dim0Padded=0, dim1Padded=0, dim2Padded=0, tileSizePadded=0;
    unsigned lenInput=0, lenInputPadded=0, lenOutput=0, lenOutputUdt=0;

    tileSizePadded = MakeDivisible<unsigned>(tileSize, CONFIG_M_AXI_WIDTH);

    if(rank==2){
        dim0=shape[0];
        dim1=shape[1]; 
        dim0Padded=MakeDivisible<unsigned>(dim0, CONFIG_M_AXI_WIDTH); 
        dim1Padded=MakeDivisible<unsigned>(dim1, CONFIG_M_AXI_WIDTH); 

        lenInput=dim0*dim1;
        lenInputPadded=dim0*dim1Padded;
        lenOutput=dim0*dim1*tileSize;
        if(tileAxis==1){
            //input: BxN ===> output: BxTxN ===> lastDim: N,(dim1)
            lenOutputUdt=dim0*tileSize*dim1Padded;
        }else if(tileAxis==2){
            //input: BxN ===> output: BxNxT ===> lastDim: T,(tileSize)
            lenOutputUdt=dim0*dim1*tileSizePadded;
        }else{
            //NYI
            assert(false);
        }
        
    }else{
        dim0=shape[0];
        dim1=shape[1]; 
        dim2=shape[2]; 
        dim0Padded=MakeDivisible<unsigned>(dim0, CONFIG_M_AXI_WIDTH); 
        dim1Padded=MakeDivisible<unsigned>(dim1, CONFIG_M_AXI_WIDTH); 
        dim2Padded=MakeDivisible<unsigned>(dim2, CONFIG_M_AXI_WIDTH); 

        lenInput=dim0*dim1*dim2;
        lenInputPadded=dim0*dim1*dim2Padded;
        lenOutput=dim0*dim1*dim2*tileSize;
        if(tileAxis==2){
            //input: BxNxD ===> output: BxNxTxD ===> lastDim: D,(dim2)
            lenOutputUdt=dim0*dim1*tileSize*dim2Padded;
        }else{
            //NYI
            assert(false);
        }
    }

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

    if(rank==2 && tileAxis==1){
        //input: BxN ===> output: BxTxN ===> lastDim: N,(dim1)
        PadTensor<CONFIG_DTYPE>(hostInputTn, hostInputTnPadded, dim0, dim1, dim1Padded);
    }else if(rank==2 && tileAxis==2){
        //input: BxN ===> output: BxNxT ===> lastDim: T,(tileSize)
        PadTensor<CONFIG_DTYPE>(hostInputTn, hostInputTnPadded, dim0, dim1, dim1Padded);
    }else if(rank==3 && tileAxis==2){
        //input: BxNxD ===> output: BxNxTxD ===> lastDim: D,(dim2)
        PadTensor<CONFIG_DTYPE>(hostInputTn, hostInputTnPadded, dim0*dim1, dim2, dim2Padded);
    }
    
    auto deviceInputTn = Pack<vecSize, CONFIG_DTYPE>(hostInputTnPadded);
    auto deviceOutputTn = Pack<vecSize, CONFIG_DTYPE>(hostUDT);

    task_tile(deviceInputTn.data(), deviceOutputTn.data(), dim0, dim1, dim2, rank, tileAxis, tileSize);

    GoldTile(
        hostInputTn.data(), 
        hostGold.data(), 
        shape,
        tileAxis,
        tileSize);

    const auto hostOutputTn = Unpack<vecSize, CONFIG_DTYPE>(deviceOutputTn);
    if(rank==2 && tileAxis==1){
        //input: BxN ===> output: BxTxN ===> lastDim: N,(dim1)
        UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUDT, dim0*tileSize, dim1Padded, dim1);
    }else if(rank==2 && tileAxis==2){
        //input: BxN ===> output: BxNxT ===> lastDim: T,(tileSize)
        UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUDT, dim0*dim1, tileSizePadded, tileSize);
    }else if(rank==3 && tileAxis==2){
        //input: BxNxD ===> output: BxNxTxD ===> lastDim: D,(dim2)
        UnpadTensor<CONFIG_DTYPE>(hostOutputTn, hostUDT, dim0*dim1*tileSize, dim2Padded, dim2);
    }
    
    bool rslt = true;

    if(rank==2 && tileAxis==1){
        //input: BxN ===> output: BxTxN ===> lastDim: N,(dim1)
        for(unsigned d0=0; d0<dim0; d0++){ 
            for(unsigned d1=0; d1<tileSize; d1++){ 
                for(unsigned d2=0; d2<dim1; d2++){ 
                    const unsigned indxCpu = d0*tileSize*dim1 + d1*dim1 + d2;
                    const unsigned indxUdt = indxCpu;
                    CONFIG_DTYPE rCpu = hostGold[indxCpu];
                    CONFIG_DTYPE rUdt = hostUDT[indxUdt];
                    CONFIG_DTYPE diff = rCpu - rUdt;
                    if(abs(diff)>1e-02){
                        printf("d0= (%03d)\td1= (%03d)\td2= (%03d)\trCPU=%f,\t\t rUDT=%f\n", 
                            d0, d1, d2, rCpu, rUdt);
                        rslt=false;
                    }
                }
            }
        }
    }else if(rank==2 && tileAxis==2){
        //input: BxN ===> output: BxNxT ===> lastDim: T,(tileSize)
        for(unsigned d0=0; d0<dim0; d0++){ 
            for(unsigned d1=0; d1<dim1; d1++){ 
                for(unsigned d2=0; d2<tileSize; d2++){ 
                    const unsigned indxCpu = d0*dim1*tileSize + d1*tileSize + d2;
                    const unsigned indxUdt = indxCpu;
                    CONFIG_DTYPE rCpu = hostGold[indxCpu];
                    CONFIG_DTYPE rUdt = hostUDT[indxUdt];
                    CONFIG_DTYPE diff = rCpu - rUdt;
                    if(abs(diff)>1e-02){
                        printf("d0= (%03d)\td1= (%03d)\td2= (%03d)\trCPU=%f,\t\t rUDT=%f\n", 
                            d0, d1, d2, rCpu, rUdt);
                        rslt=false;
                    }
                }
            }
        }
    }else if(rank==3 && tileAxis==2){
        //input: BxNxD ===> output: BxNxTxD ===> lastDim: D,(dim2)
        for(unsigned d0=0; d0<dim0; d0++){ 
            for(unsigned d1=0; d1<dim1; d1++){ 
                for(unsigned d2=0; d2<tileSize; d2++){ 
                    for(unsigned d3=0; d3<dim2; d3++){ 
                        const unsigned indxCpu = d0*dim1*tileSize*dim2 + 
                                                 d1*tileSize*dim2 + 
                                                 d2*dim2 + 
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
    }else{
        assert(false);
    }

    std::cout<<std::endl;

    if(rslt){
        std::cout<<"Test \""<<testName<<"\" is successfully verified."<<std::endl;
    }

    return (rslt)? 0 : 1;
}

int main(int argc, char **argv) {
    int rslt = 0;
    rslt += TestTile<16>("TileRank2Axis2(BxN to BxNxT)", {2,2}, 2, 3);
    rslt += TestTile<16>("TileRank2Axis2(BxN to BxNxT)", {2,5}, 2, 7);
    rslt += TestTile<16>("TileRank2Axis1(BxN to BxTxN)", {2,2}, 1, 3);
    rslt += TestTile<16>("TileRank3Axis2(BxNxD to BxNxTxD)", {2,2,2}, 2, 3);
    return rslt;
}

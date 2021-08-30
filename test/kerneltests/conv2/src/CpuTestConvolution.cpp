/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @date      June 2017 
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "PaddingCpu.h"
#include "Utility.h"
#include "Conv2D.h"
#include "Conv2Helper.h"
#include "xilinx/config.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

using namespace ConfigTaskConv2;

extern "C"
void task_conv2_1x1_direct(
    MemoryPackK_t const a[],
    MemoryPackM_t const b[],
    MemoryPackM_t const e[],
    MemoryPackM_t c[],
    const unsigned size_n, 
    const unsigned size_k,
    const unsigned size_m);

int main(int argc, char **argv) {

    unsigned conv_b = 1;
    unsigned conv_n = 1;
    unsigned conv_k = 256;
    unsigned conv_din = 6;
    unsigned conv_dout = ConfigTaskConv2::kOuterTileSizeM;

    if (argc < 6 || argc > 6) {
        std::cout << "Usage: ./TestSimulation Conv_B Conv_N Conv_K Conv_Din Conv_Dout" << std::endl;
        std::cout << "Running with the default parameters(B,N,K,Din,Dout)=("<< 
                conv_b << "," <<
                conv_n << "," <<
                conv_k << "," <<  
                conv_din << "," <<
                conv_dout << "," <<
                ")"<<std::endl;
    }else{
        conv_b = std::stoul(argv[1]);
        conv_n = std::stoul(argv[2]);
        conv_k = std::stoul(argv[3]);
        conv_din = std::stoul(argv[4]);
        conv_dout = std::stoul(argv[5]);
    }

    std::cout << "Convolution Shapes: Data(BxNxKxD1), Weight(D1xD2) :   " <<
            conv_b << "x" << conv_n << "x" << conv_k << "x" << conv_din << "  " <<
            conv_din << "x" << conv_dout << std::endl;

    const unsigned size_n = conv_b*conv_n*conv_k;
    const unsigned size_k = conv_din;
    const unsigned size_m = conv_dout;

    std::cout << "[N, K, M] = [" << size_n << ", " << size_k << ", " << size_m << "]" << std::endl;

    /*if (size_k % kMemoryWidthK != 0) {
    std::cerr << "K must be divisable by memory width." << std::endl;
    return 1;
    }*/

    if (size_m % kMemoryWidthM != 0) {
        std::cerr << "M must be divisable by memory width." << std::endl;
        return 1;
    }
    if (size_n % kOuterTileSizeN != 0) {
        std::cerr << "N must be divisable by the outer tile size in N."
        << std::endl;
        return 1;
    }
    if (size_m % kOuterTileSizeM != 0) {
        std::cerr << "M must be divisable by the outer tile size in M" << std::endl;
        return 1;
    }

    const auto size_k_padded = DivCeil<unsigned>(size_k, kTransposeWidth)*kTransposeWidth;

    std::vector<CONFIG_DTYPE> a(size_n * size_k); //input
    std::vector<CONFIG_DTYPE> b(size_k * size_m); //weight
    std::vector<CONFIG_DTYPE> e(size_m); //bias
    std::vector<CONFIG_DTYPE> cReference(size_n * size_m, 0);
    std::vector<CONFIG_DTYPE> cReferenceConv2(size_n * size_m, 0);

    std::default_random_engine rng(kSeed);
    typename std::conditional<
        std::is_integral<CONFIG_DTYPE>::value, std::uniform_int_distribution<unsigned long>,
        std::uniform_real_distribution<double>>::type dist(1, 10);

    for_each(a.begin(), a.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });
    for_each(b.begin(), b.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });
    for_each(e.begin(), e.end(),
        [&dist, &rng](CONFIG_DTYPE &in) { in = CONFIG_DTYPE(dist(rng)); });

    std::vector<CONFIG_DTYPE> aPadded(size_n * size_k_padded);
    PadTensor<CONFIG_DTYPE>(a, aPadded, size_n, size_k, DivCeil<unsigned>(size_k, kTransposeWidth)*kTransposeWidth);

    const auto aKernel = Pack<kMemoryWidthA, CONFIG_DTYPE>(aPadded);
    const auto bKernel = Pack<kMemoryWidthM, CONFIG_DTYPE>(b);
    const auto eKernel = Pack<kMemoryWidthM, CONFIG_DTYPE>(e);
    auto cKernel = Pack<kMemoryWidthM, CONFIG_DTYPE>(cReference);

    ReferenceImplementation(a.data(), b.data(), cReference.data(), size_n, size_k, size_m);
    Conv2Kernel1x1CPU<CONFIG_DTYPE >(
        a.data(), b.data(), e.data(), cReferenceConv2.data(), 
        conv_b, conv_n, conv_k, conv_din, conv_dout);

    std::cout << "Running simulation...\n" << std::flush;

    task_conv2_1x1_direct(
        aKernel.data(), bKernel.data(), 
        eKernel.data(), cKernel.data(),
        size_n, size_k, size_m);
    std::cout << "Verifying results...\n" << std::flush;

    const auto cTest = Unpack<kMemoryWidthM, CONFIG_DTYPE>(cKernel);

    for (unsigned i = 0; i < size_n; ++i) {
        for (unsigned j = 0; j < size_m; ++j) {
            const auto testVal = make_signed<CONFIG_DTYPE>(cTest[i * size_m + j]);
            const auto refVal = make_signed<CONFIG_DTYPE>(cReference[i * size_m + j]);
            const auto refValConv2 = make_signed<CONFIG_DTYPE>(cReferenceConv2[i * size_m + j]);
            //const CONFIG_DTYPE diff = abs(testVal - refVal);
            const CONFIG_DTYPE diff2 = abs(testVal - refValConv2);
            /*if (diff > static_cast<CONFIG_DTYPE>(1e-3)) {
            std::cerr << "Mismatch detected(Kernel vs. CPU MM) at (" << i << ", " << j
            << "): " << testVal << " vs. " << refVal << "\n";
            return 1;
            }*/
            if (diff2 /*/ refValConv2*/ > static_cast<CONFIG_DTYPE>(1e-2)) {
                std::cerr << "Mismatch detected(Kernel vs. CPU Conv2) at (" << i << ", " << j
                    << "): " << testVal << " vs. " << refValConv2 << "\n";
                return 2;
            }
        }
    }
    //std::cout << "Matrix-matrix multiplication successfully verified.\n";
    std::cout << "Conv2D 1x1 successfully verified.\n";

    return 0;
}

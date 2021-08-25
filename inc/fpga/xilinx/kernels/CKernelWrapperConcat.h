#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "fpga/xilinx/CKernelArg.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperConcat: public CKernelWrapper{
 public:
  CTensorXil<float>* EnqueueKernelLaunch(CTensorXil<float> *inputTn1, CTensorXil<float> *inputTn2, unsigned concatDim){
    if(inputTn1->GetRank()!=4){
      throw std::runtime_error(CStringFormatter()<<__func__<<": Bad input tensor rank.");
    }
    if(inputTn2->GetRank()!=4){
      throw std::runtime_error(CStringFormatter()<<__func__<<": Bad input tensor rank.");
    }
    if(concatDim!=3){
      throw std::runtime_error(CStringFormatter()<<__func__<<": Only concatDim=3 is implemented.");
    }

    const auto shape1 = inputTn1->GetShape();
    const auto shape2 = inputTn2->GetShape();
    assert(shape1[0]==shape2[0]);
    assert(shape1[1]==shape2[1]);
    assert(shape1[2]==shape2[2]);

    const unsigned dim0 = shape1[0];
    const unsigned dim1 = shape1[1];
    const unsigned dim2 = shape1[2];
    const unsigned dimA3 = shape1[3];
    const unsigned dimB3 = shape2[3];
    const unsigned dimR3 = dimA3+dimB3;
    const std::vector<unsigned> shapeOut = {dim0, dim1, dim2, dimR3};
    CTensorXil *outputTn = new CTensorXil(GetXilInfo(), shapeOut, GetBankIndex());

    ResetArgCounter();
    GetKernel()->setArg(ArgCounter(), inputTn1->GetDeviceBuffer());
    GetKernel()->setArg(ArgCounter(), inputTn2->GetDeviceBuffer());
    GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer());
    GetKernel()->setArg(ArgCounter(), dim0);
    GetKernel()->setArg(ArgCounter(), dim1);
    GetKernel()->setArg(ArgCounter(), dim2);
    GetKernel()->setArg(ArgCounter(), dimA3);
    GetKernel()->setArg(ArgCounter(), dimB3);
    GetKernel()->setArg(ArgCounter(), concatDim);

    GetXilInfo()->GetQueue()->enqueueTask(
        *GetKernel(),
        {*inputTn1->GetEventPtr(), *inputTn2.GetEventPtr()},
        outputTn->GetEventPtr()
    );


    outputTn->GetEventPtr()->setCallback(CL_COMPLETE, EventCallback, ) ???????????????????
  }
};

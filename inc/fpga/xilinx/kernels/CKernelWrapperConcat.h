#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperConcat: public CKernelWrapper{
 public:
  CKernelWrapperConcat(
      std::string taskName,
      std::string fileName,
      CXilinxInfo *xilInfo,
      unsigned bankInputTn1,
      unsigned bankInputTn2,
      unsigned bankOutputTn,
      std::string path,
      bool isDisabled,
      bool profileOcl
      ):CKernelWrapper(
      taskName,
      fileName,
      xilInfo,
      path,
      isDisabled,
      profileOcl){

    m_uBankInputTn1=bankInputTn1;
    m_uBankInputTn2=bankInputTn2;
    m_uBankOutputTn=bankOutputTn;

  }

  CTensorXil<float>* EnqueueKernelLaunch(unsigned parentLayerId, CTensorXil<float> *inputTn1, CTensorXil<float> *inputTn2, unsigned concatDim){
    if(inputTn1->GetRank()!=4){
      throw std::runtime_error(CStringFormatter()<<__func__<<": Bad input tensor rank.");
    }
    if(inputTn2->GetRank()!=4){
      throw std::runtime_error(CStringFormatter()<<__func__<<": Bad input tensor rank.");
    }
    if(concatDim!=3){
      throw std::runtime_error(CStringFormatter()<<__func__<<": Only concatDim=3 is implemented.");
    }

    auto *xinputTn1 = inputTn1->CloneIfNeededToBank(m_uBankInputTn1);
    auto *xinputTn2 = inputTn1->CloneIfNeededToBank(m_uBankInputTn2);

    const auto shape1 = xinputTn1->GetShape();
    const auto shape2 = xinputTn2->GetShape();
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
    auto *outputTn = new CTensorXil<float>(GetXilInfo(), shapeOut, false, m_uBankOutputTn);

    ResetArgCounter();
    GetKernel()->setArg(ArgCounter(), xinputTn1->GetDeviceBuffer());
    GetKernel()->setArg(ArgCounter(), xinputTn2->GetDeviceBuffer());
    GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer());
    GetKernel()->setArg(ArgCounter(), dim0);
    GetKernel()->setArg(ArgCounter(), dim1);
    GetKernel()->setArg(ArgCounter(), dim2);
    GetKernel()->setArg(ArgCounter(), dimA3);
    GetKernel()->setArg(ArgCounter(), dimB3);
    GetKernel()->setArg(ArgCounter(), concatDim);

    std::vector<cl::Event> dependencies;
    dependencies.push_back(*inputTn1->GetEventPtr());
    dependencies.push_back(*inputTn2->GetEventPtr());

    GetXilInfo()->GetQueue()->enqueueTask(
        *GetKernel(),
        &dependencies,
        outputTn->GetEventPtr()
    );


    m_ptrCallBackData.get()->profileKernel = GetProfileOclEnabled();
    m_ptrCallBackData.get()->parentLayerId = parentLayerId;
    m_ptrCallBackData.get()->classPtr = this;


    outputTn->GetEventPtr()->setCallback(CL_COMPLETE, &EventCallback, m_ptrCallBackData.get());
  }

 private:
  unsigned m_uBankInputTn1;
  unsigned m_uBankInputTn2;
  unsigned m_uBankOutputTn;
};

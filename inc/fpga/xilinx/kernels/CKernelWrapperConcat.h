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

  CTensorBasePtr EnqueueKernelLaunch(unsigned parentLayerId, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, unsigned concatDim){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks

    ConditionCheck(inputTn1->GetRank()==4, "Bad input tensor rank.");
    ConditionCheck(inputTn2->GetRank()==4, "Bad input tensor rank.");
    ConditionCheck(concatDim==3, "Only concatDim==3 is supported.");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn1 = std::static_pointer_cast<CTensorXil<float>>(inputTn1);
    auto pInputTn2 = std::static_pointer_cast<CTensorXil<float>>(inputTn2);
    auto xInputTn1 = pInputTn1->CloneIfNeededToBank(m_uBankInputTn1);
    auto xInputTn2 = pInputTn2->CloneIfNeededToBank(m_uBankInputTn2);

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch

    const auto shape1 = xInputTn1->GetShape();
    const auto shape2 = xInputTn2->GetShape();
    ConditionCheck(shape1[0]==shape2[0], "Unequal shape[0]'s.");
    ConditionCheck(shape1[1]==shape2[1], "Unequal shape[1]'s.");
    ConditionCheck(shape1[2]==shape2[2], "Unequal shape[2]'s.");

    const unsigned dim0 = shape1[0];
    const unsigned dim1 = shape1[1];
    const unsigned dim2 = shape1[2];
    const unsigned dimA3 = shape1[3];
    const unsigned dimB3 = shape2[3];
    const unsigned dimR3 = dimA3+dimB3;

    if(dimR3 >= CONFIG_M_AXI_WIDTH){
      if(dimA3 % CONFIG_M_AXI_WIDTH !=0 || dimB3 % CONFIG_M_AXI_WIDTH !=0){
        ThrowException("for dimR3>=CONFIG_M_AXI_WIDTH, only tensors with dim3%CONFIG_M_AXI_WIDTH=0 are supported.");
      }
    }

    const std::vector<unsigned> shapeOut = {dim0, dim1, dim2, dimR3};
    CTensorXilPtr<float> outputTn(new CTensorXil<float>(GetXilInfo(), shapeOut, false, m_uBankOutputTn));

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn1->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn2->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim0));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim1));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim2));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dimA3));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dimB3));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), concatDim));

    std::vector<cl::Event> dependencies;

    // Double check to make sure that bank-crossed tensors are used here.
    dependencies.push_back(*xInputTn1->GetEventPtr());
    dependencies.push_back(*xInputTn1->GetEventPtr());

    GetXilInfo()->GetQueue()->enqueueTask(
        *GetKernel(),
        &dependencies,
        outputTn->GetEventPtr()
    );

    // -----------------------------------------------------------------------------------------------------------------
    // #. Callbacks And Book-keepings
    auto *callBackDataPtr = GenerateAndStoreCallBackData(this, parentLayerId);
    outputTn->GetEventPtr()->setCallback(CL_COMPLETE, &EventCallback, callBackDataPtr);

    // WARNING: Always store:
    //  - the raw input tensors
    //  - the bank-crossed versions of the input tensors
    //  - the output tensors
    // Not storing raw input tensors could allow a tensor to be released
    // before its async bank-crossing operation is executed, resulting in data loss and/or fatal crash.
    StoreBookKeepingEntry({inputTn1, inputTn2, xInputTn1, xInputTn2, outputTn});

    // -----------------------------------------------------------------------------------------------------------------
    // #. Returning Part
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn1;
  unsigned m_uBankInputTn2;
  unsigned m_uBankOutputTn;
};

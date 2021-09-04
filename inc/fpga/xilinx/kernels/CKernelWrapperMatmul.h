#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperMatmul: public CKernelWrapper{
 public:
  CKernelWrapperMatmul(
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

  CTensorBasePtr EnqueueKernelLaunch(unsigned parentLayerId, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks

    ConditionCheck(inputTn1->GetRank()==inputTn2->GetRank(), "The inputs are of unequal ranks.");
    ConditionCheck(inputTn1->GetRank()==3 || inputTn1->GetRank()==2, "Bad inputTn1 tensor rank.");
    ConditionCheck(inputTn2->GetRank()==3 || inputTn2->GetRank()==2, "Bad inputTn2 tensor rank.");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn1 = std::static_pointer_cast<CTensorXil<float>>(inputTn1);
    auto pInputTn2 = std::static_pointer_cast<CTensorXil<float>>(inputTn2);
    auto xInputTn1 = pInputTn1->CloneIfNeededToBank(m_uBankInputTn1);
    auto xInputTn2 = pInputTn2->CloneIfNeededToBank(m_uBankInputTn2);

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    unsigned diff = xInputTn1->ExpandDimZeroToRank(3);
    xInputTn2->ExpandDimZeroToRank(3);


    const auto shape1 = xInputTn1->GetShape();
    const auto shape2 = xInputTn2->GetShape();

    auto dim0A = shape1[0]; // batch size
    auto dim1A = shape1[1]; // height of matrix ,N
    auto dim2A = shape1[2]; // width of matrix  ,K
    auto dim0B = shape2[0]; // batch size
    auto dim1B = shape2[1]; // height of matrix ,K
    auto dim2B = shape2[2]; // width of matrix  ,M

    // Width of A should be equal to the Height of B. (dim2A = dim1B)
    ConditionCheck(dim0A == dim0B,"Unequal batch sizes.");
    ConditionCheck(dim2A == dim1B, "Unequal shape1[2] and shape2[1].");

    CTensorXilPtr<float> outputTn(new CTensorXil<float>(GetXilInfo(), {dim0A,dim1A,dim2B}, false, m_uBankOutputTn));

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn1->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn2->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim0A));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim1A));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim2A));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim2B));

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
    xInputTn1->SqueezeDimZeroTimesTry(diff);
    xInputTn2->SqueezeDimZeroTimesTry(diff);
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn1;
  unsigned m_uBankInputTn2;
  unsigned m_uBankOutputTn;
};

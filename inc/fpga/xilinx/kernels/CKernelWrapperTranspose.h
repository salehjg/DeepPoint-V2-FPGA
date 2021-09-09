#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperTranspose: public CKernelWrapper{
 public:
  CKernelWrapperTranspose(
      std::string taskName,
      std::string fileName,
      CXilinxInfo *xilInfo,
      unsigned bankInputTn,
      unsigned bankOutputTn,
      std::string path,
      bool isDisabled,
      bool profileOcl,
      bool logMemBankCrossings
  ):CKernelWrapper(
      taskName,
      fileName,
      xilInfo,
      path,
      isDisabled,
      profileOcl,
      logMemBankCrossings){

    m_uBankInputTn=bankInputTn;
    m_uBankOutputTn=bankOutputTn;

  }

  CTensorBasePtr EnqueueKernelLaunch(unsigned parentLayerId, CTensorBasePtr inputTn){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks
    ConditionCheck(inputTn->GetRank()==2 || inputTn->GetRank()==3,"Unsupported tensor rank.");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn = std::static_pointer_cast<CTensorXil<float>>(inputTn);
    auto xInputTn = pInputTn->CloneIfNeededToBank(m_uBankInputTn);
    if(m_bLogMemBankCrossings) m_vMemBankCrossings.push_back("abs("+ (pInputTn)->GetTensorTag() +"-transpose_in)");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    unsigned diff = xInputTn->ExpandDimZeroToRank(3);
    auto shape = xInputTn->GetShape();
    auto dim0 = shape[0];
    auto dim1 = shape[1];
    auto dim2 = shape[2];

    ConditionCheck(dim1%(ConfigTaskTranspose::PipeDepth1)==0, "Incompatible dim1 for ConfigTaskTranspose::PipeDepth1");
    ConditionCheck(dim2%(ConfigTaskTranspose::PipeDepth2)==0, "Incompatible dim2 for ConfigTaskTranspose::PipeDepth2");

    CTensorXilPtr<float> outputTn(new CTensorXil<float>(GetXilInfo(), {dim0,dim2,dim1}, false, m_uBankOutputTn));

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim0));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim1));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim2));

    std::vector<cl::Event> dependencies;

    // Double check to make sure that bank-crossed tensors are used here.
    dependencies.push_back(*xInputTn->GetEventPtr());

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
    StoreBookKeepingEntry({inputTn, xInputTn, outputTn});

    // -----------------------------------------------------------------------------------------------------------------
    // #. Returning Part
    xInputTn->SqueezeDimZeroTimesTry(diff);
    if(m_bLogMemBankCrossings) outputTn->SetTensorTag("transpose_out");
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn;
  unsigned m_uBankOutputTn;
};

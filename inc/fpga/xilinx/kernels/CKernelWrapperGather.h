#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperGather: public CKernelWrapper{
 public:
  CKernelWrapperGather(
      std::string taskName,
      std::string fileName,
      CXilinxInfo *xilInfo,
      unsigned bankInputTn,
      unsigned bankIndicesTn,
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

    m_uBankInputTn=bankInputTn;
    m_uBankIndicesTn=bankIndicesTn;
    m_uBankOutputTn=bankOutputTn;
  }

  CTensorBasePtr EnqueueKernelLaunch(unsigned parentLayerId, CTensorBasePtr inputTn, CTensorBasePtr indicesTn, unsigned indicesOfAxis){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks
    ConditionCheck(inputTn->GetRank()==3, "inputTn is required to be of rank 3.");
    ConditionCheck(indicesTn->GetRank()==3, "indicesTn is required to be of rank 3.");
    ConditionCheck(inputTn->GetShape()[0]==indicesTn->GetShape()[0], "Incompatible shapes.");
    ConditionCheck(inputTn->GetShape()[1]==indicesTn->GetShape()[1], "Incompatible shapes.");
    ConditionCheck(indicesOfAxis==1, "Unsupported indicesOfAxis.");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn = std::static_pointer_cast<CTensorXil<float>>(inputTn);
    auto pIndicesTn = std::static_pointer_cast<CTensorXil<unsigned>>(indicesTn);
    auto xInputTn = pInputTn->CloneIfNeededToBank(m_uBankInputTn);
    auto xIndicesTn = pIndicesTn->CloneIfNeededToBank(m_uBankIndicesTn);

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    unsigned B,N,D,K;
    auto shape = xInputTn->GetShape();
    B = shape[0];
    N = shape[1];
    D = shape[2];
    K = xIndicesTn->GetShape()[2];

    CTensorXilPtr<float> outputTn(new CTensorXil<float>(GetXilInfo(), {B,N,K,D}, false, m_uBankOutputTn));

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xIndicesTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), indicesOfAxis));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), B));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), N));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), D));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), B));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), N));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), K));

    std::vector<cl::Event> dependencies;

    // Double check to make sure that bank-crossed tensors are used here.
    dependencies.push_back(*xInputTn->GetEventPtr());
    dependencies.push_back(*xIndicesTn->GetEventPtr());

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
    StoreBookKeepingEntry({inputTn, xInputTn, indicesTn, xIndicesTn, outputTn});

    // -----------------------------------------------------------------------------------------------------------------
    // #. Returning Part
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn;
  unsigned m_uBankIndicesTn;
  unsigned m_uBankOutputTn;
};

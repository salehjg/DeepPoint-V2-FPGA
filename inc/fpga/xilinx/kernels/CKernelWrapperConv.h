#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperConv: public CKernelWrapper{
 public:
  CKernelWrapperConv(
      std::string taskName,
      std::string fileName,
      CXilinxInfo *xilInfo,
      unsigned bankInputTn,
      unsigned bankWeightTn,
      unsigned bankBiasTn,
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
    m_uBankWeightTn=bankWeightTn;
    m_uBankBiasTn=bankBiasTn;
    m_uBankOutputTn=bankOutputTn;

  }

  CTensorBasePtr EnqueueKernelLaunch(
      unsigned parentLayerId,
      CTensorBasePtr inputTn,
      CTensorBasePtr weightTn,
      CTensorBasePtr biasTn,
      unsigned B,
      unsigned N,
      unsigned K,
      unsigned D1,
      unsigned D2Padded
      ){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn = std::static_pointer_cast<CTensorXil<float>>(inputTn);
    auto xInputTn = pInputTn->CloneIfNeededToBank(m_uBankInputTn);
    if(m_bLogMemBankCrossings) m_vMemBankCrossings.push_back("abs("+ (pInputTn)->GetTensorTag() +"-conv_in)");

    auto pWeightTn = std::static_pointer_cast<CTensorXil<float>>(weightTn);
    auto xWeightTn = pWeightTn->CloneIfNeededToBank(m_uBankWeightTn);
    if(m_bLogMemBankCrossings) m_vMemBankCrossings.push_back("abs("+ (pWeightTn)->GetTensorTag() +"-conv_w)");

    auto pBiasTn = std::static_pointer_cast<CTensorXil<float>>(biasTn);
    auto xBiasTn = pBiasTn->CloneIfNeededToBank(m_uBankBiasTn);
    if(m_bLogMemBankCrossings) m_vMemBankCrossings.push_back("abs("+ (pBiasTn)->GetTensorTag() +"-conv_b)");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    CTensorXilPtr<float> outputTn(
        new CTensorXil<float>(GetXilInfo(), {B,N,K,D2Padded}, false, m_uBankOutputTn));

    const unsigned sizeN = B*N*K;
    const unsigned sizeK = D1; //Should be the original shape, not the padded one.
    const unsigned sizeM = D2Padded;

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xWeightTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xBiasTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), sizeN));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), sizeK));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), sizeM));

    std::vector<cl::Event> dependencies;

    // Double check to make sure that bank-crossed tensors are used here.
    dependencies.push_back(*xInputTn->GetEventPtr());
    dependencies.push_back(*xWeightTn->GetEventPtr());
    dependencies.push_back(*xBiasTn->GetEventPtr());

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
    StoreBookKeepingEntry({
      inputTn, xInputTn,
      weightTn, xWeightTn,
      biasTn, xBiasTn,
      outputTn});

    // -----------------------------------------------------------------------------------------------------------------
    // #. Returning Part
    if(m_bLogMemBankCrossings) outputTn->SetTensorTag("conv_out");
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn;
  unsigned m_uBankWeightTn;
  unsigned m_uBankBiasTn;
  unsigned m_uBankOutputTn;
};

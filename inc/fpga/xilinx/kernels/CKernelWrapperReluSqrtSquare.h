#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperReluSqrtSquare: public CKernelWrapper{
 public:
  CKernelWrapperReluSqrtSquare(
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

  CTensorBasePtr EnqueueKernelLaunch(unsigned parentLayerId, CTensorBasePtr inputTn, bool runRelu, bool runSqrt, bool runSquare){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks

    ConditionCheck(
        (runRelu&&!runSqrt&&!runSquare)||(!runRelu&&runSqrt&&!runSquare)||(!runRelu&&!runSqrt&&runSquare),
        "Only one of the modes is allowed to be selected.");
    ConditionCheck(inputTn->GetLen()!=0, "The input tensor is of length zero!");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn = std::static_pointer_cast<CTensorXil<float>>(inputTn);
    auto xInputTn = pInputTn->CloneIfNeededToBank(m_uBankInputTn);
    if(m_bLogMemBankCrossings) m_vMemBankCrossings.push_back("abs("+ (pInputTn)->GetTensorTag() +"-relusqrtsquare_in)");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    const auto shape = xInputTn->GetShape();
    const unsigned mode = runRelu?ConfigTaskReluSqrtSquare::ModeRelu:
                          runSqrt?ConfigTaskReluSqrtSquare::ModeSqrt:
                          runSquare?ConfigTaskReluSqrtSquare::ModeSquare:
                          100;

    CTensorXilPtr<float> outputTn(new CTensorXil<float>(GetXilInfo(), shape, false, m_uBankOutputTn));
    auto len = (cl_uint)(xInputTn->GetVectorCountPadded());

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), len));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), mode));

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
    if(m_bLogMemBankCrossings) outputTn->SetTensorTag("relusqrtsquare_out");
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn;
  unsigned m_uBankOutputTn;
};

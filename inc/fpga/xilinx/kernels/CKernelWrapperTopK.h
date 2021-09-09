#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperTopK: public CKernelWrapper{
 public:
  CKernelWrapperTopK(
      std::string taskName,
      std::string fileName,
      CXilinxInfo *xilInfo,
      unsigned bankInputTn,
      unsigned bankOutputTn,
      unsigned maxSliceLen,
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
    m_uMaxSliceLen=maxSliceLen;

  }

  CTensorBasePtr EnqueueKernelLaunch(unsigned parentLayerId, CTensorBasePtr inputTn, unsigned axis, unsigned k){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks
    {
      const auto _lastDim = inputTn->GetShape()[2];
      ConditionCheck(inputTn->GetRank()==3, "Only input tensors of rank 3 are supported.");
      ConditionCheck(_lastDim%CONFIG_M_AXI_WIDTH==0, "shape[2] should be divisible by AXI's width.");
      ConditionCheck(axis==2, "Only axis=2 is supported.");
      ConditionCheck(_lastDim==m_uMaxSliceLen, "shape[2] should be equal to MaxSliceLen.");
    }

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn = std::static_pointer_cast<CTensorXil<float>>(inputTn);
    auto xInputTn = pInputTn->CloneIfNeededToBank(m_uBankInputTn);
    if(m_bLogMemBankCrossings) m_vMemBankCrossings.push_back("abs("+ (pInputTn)->GetTensorTag() +"-topk_in)");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    const auto shape = xInputTn->GetShape();
    auto outputShape = xInputTn->GetShape();
    outputShape[2]=k;
    const unsigned batchSize = outputShape[0]*outputShape[1];
    const auto vecsPerSlice = DivCeil<unsigned>(shape[2], CONFIG_M_AXI_WIDTH);
    const auto vecsPerOutputSlice = DivCeil<unsigned>(k, CONFIG_M_AXI_WIDTH);
    const auto _dim2 = shape[2];
    CTensorXilPtr<unsigned> outputTn(new CTensorXil<unsigned>(GetXilInfo(), outputShape, false, m_uBankOutputTn));

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), batchSize));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), _dim2));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), k));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), vecsPerSlice));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), vecsPerOutputSlice));

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
    if(m_bLogMemBankCrossings) outputTn->SetTensorTag("topk_out");
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn;
  unsigned m_uBankOutputTn;
  unsigned m_uMaxSliceLen;
};

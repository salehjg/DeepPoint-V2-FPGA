#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperPadUnpad: public CKernelWrapper{
 public:
  CKernelWrapperPadUnpad(
      std::string taskName,
      std::string fileName,
      CXilinxInfo *xilInfo,
      unsigned bankInputTn,
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
    m_uBankOutputTn=bankOutputTn;
  }

  CTensorBasePtr EnqueueKernelLaunch(
      unsigned parentLayerId,
      CTensorBasePtr inputTn,
      bool pad,
      bool unpad,
      unsigned lastDimPadded,
      unsigned lastDimUnpadded){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks
    {
      const auto shape = inputTn->GetShape();
      ConditionCheck((pad && !unpad) || (!pad && unpad), "One of the args should be selected (pad or unpad).")
      ConditionCheck(
          (pad && (lastDimPadded >= shape.back())) ||
              (unpad && (lastDimUnpadded <= shape.back())),
          "One of the args should be selected (pad or unpad)."
      );
      if (pad) {
        ConditionCheck(lastDimPadded >= CONFIG_M_AXI_WIDTH,
                       "Sub-vec padding/unpadding is disabled in the kernel.");
        ConditionCheck(lastDimPadded % CONFIG_M_AXI_WIDTH == 0,
                       "lastDimPadded should be divisible by AXI's width.");
        ConditionCheck(shape.back() >= CONFIG_M_AXI_WIDTH,
                       "shape[-1] should be greater or equal to AXI's width.");
      }
      if (unpad) {
        ConditionCheck(shape.back() % CONFIG_M_AXI_WIDTH == 0,
                       "shape[-1] should be divisible by AXI's width.");
        ConditionCheck(lastDimUnpadded >= CONFIG_M_AXI_WIDTH,
                       "lastDimUnpadded should be greater or equal to AXI's width.");
        ConditionCheck(lastDimUnpadded % CONFIG_M_AXI_WIDTH == 0,
                       "lastDimUnpadded should be divisible by AXI's width.");
        ConditionCheck(lastDimUnpadded < shape.back(),
                       "lastDimUnpadded should be less than shape[-1].");
      }
    }
    
    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn = std::static_pointer_cast<CTensorXil<float>>(inputTn);
    auto xInputTn = pInputTn->CloneIfNeededToBank(m_uBankInputTn);

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    auto shape = xInputTn->GetShape();
    const auto rank = xInputTn->GetRank();

    unsigned dim0, dim1, lcm, _gcd, mode;
    CTensorXilPtr<float> outputTn;

    if(rank!=1){
      dim0=1;
      for(int i=0; i<rank-1; i++){
        dim0*=shape[i];
      }
      dim1=shape[rank-1];
    }else{
      dim0 = 1;
      dim1 = shape[0];
    }

    if(pad){
      if(shape[rank-1]<CONFIG_M_AXI_WIDTH){
        //sub-vector padding
        _gcd = std::__gcd(dim1, CONFIG_M_AXI_WIDTH);
        lcm = (dim1*CONFIG_M_AXI_WIDTH)/(_gcd);
      }else{
        lcm=0;
      }
      shape[rank-1] = lastDimPadded;
      outputTn = CTensorXilPtr<float>(new CTensorXil<float>(GetXilInfo(), shape, false, m_uBankOutputTn));
      mode=1;

    }else if(unpad){
      shape[rank-1] = lastDimUnpadded;
      outputTn = CTensorXilPtr<float>(new CTensorXil<float>(GetXilInfo(), shape, false, m_uBankOutputTn));
      mode=2;

    }else{
      assert(false); // sth's wrong
    }

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), mode));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim0));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim1));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), lastDimPadded));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), lcm));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), lastDimUnpadded));

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
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn;
  unsigned m_uBankOutputTn;
};

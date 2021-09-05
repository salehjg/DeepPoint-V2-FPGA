#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperBasicOps: public CKernelWrapper{
 public:
  CKernelWrapperBasicOps(
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

  CTensorBasePtr EnqueueKernelLaunch(unsigned parentLayerId, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, BASIC_OPS mode, bool useScalar, float scalar){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks

    if(!useScalar) ConditionCheck(inputTn1->GetRank()>=inputTn2->GetRank(), "The first input tensor's rank cannot be smaller than the second's.");
    ConditionCheck(inputTn1->GetRank()>=1 && inputTn1->GetRank()<=4, "Bad inputTn1 tensor rank.");
    if(!useScalar) ConditionCheck(inputTn2->GetRank()>=1 && inputTn2->GetRank()<=4, "Bad inputTn2 tensor rank.");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn1 = std::static_pointer_cast<CTensorXil<float>>(inputTn1);
    auto xInputTn1 = pInputTn1->CloneIfNeededToBank(m_uBankInputTn1);

    CTensorXilPtr<float> pInputTn2, xInputTn2;
    pInputTn2 = nullptr;
    xInputTn2 = nullptr;
    if(!useScalar){
      pInputTn2 = std::static_pointer_cast<CTensorXil<float>>(inputTn2);
      xInputTn2 = pInputTn2->CloneIfNeededToBank(m_uBankInputTn2);
    }

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    unsigned diff = xInputTn1->ExpandDimZeroToRank(4);

    const auto shape1 = xInputTn1->GetShape();
    const auto shape2 = (!useScalar) ? xInputTn2->GetShape() : std::vector<unsigned>{1};

    auto dim0A = shape1[0];
    auto dim1A = shape1[1];
    auto dim2A = shape1[2];
    auto dim3A = shape1[3];
    unsigned dim0B, dim1B, dim2B, dim3B;
    unsigned rank1 = xInputTn1->GetRank();
    unsigned rank2;
    if(useScalar){
      rank2 = 1;
    }else{
      rank2 = xInputTn2->GetRank();
    }

    if(rank2==4){
      dim0B=shape2[0];
      dim1B=shape2[1];
      dim2B=shape2[2];
      dim3B=shape2[3];
    }
    if(rank2==3){
      dim0B=0;
      dim1B=shape2[0];
      dim2B=shape2[1];
      dim3B=shape2[2];
    }
    if(rank2==2){
      dim0B=0;
      dim1B=0;
      dim2B=shape2[0];
      dim3B=shape2[1];
    }
    if(rank2==1){
      dim0B=0;
      dim1B=0;
      dim2B=0;
      dim3B=shape2[0];
    }

    int operationMode = mode==BASIC_OPS::ADD ? 0 :
                        mode==BASIC_OPS::SUB ? 1 :
                        mode==BASIC_OPS::MUL_ELEMENTWISE ? 2 :
                        3;

    CTensorXilPtr<float> outputTn(new CTensorXil<float>(GetXilInfo(), shape1, false, m_uBankOutputTn));

    cl_int stat;
    ResetArgCounter();

    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn1->GetDeviceBuffer()));
    if(!useScalar){
      OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn2->GetDeviceBuffer()));
    }
    else {
      OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn1->GetDeviceBuffer())); // dummy buffer
    }
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim0A));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim1A));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim2A));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim3A));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim0B));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim1B));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim2B));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim3B));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), rank1));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), rank2));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), operationMode));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), scalar));

    std::vector<cl::Event> dependencies;

    // Double check to make sure that bank-crossed tensors are used here.
    dependencies.push_back(*xInputTn1->GetEventPtr());
    if(!useScalar)dependencies.push_back(*xInputTn2->GetEventPtr());

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
    if(useScalar){
      StoreBookKeepingEntry({inputTn1, xInputTn1, outputTn});
    }else{
      StoreBookKeepingEntry({inputTn1, inputTn2, xInputTn1, xInputTn2, outputTn});
    }


    // -----------------------------------------------------------------------------------------------------------------
    // #. Returning Part
    xInputTn1->SqueezeDimZeroTimesTry(diff);
    outputTn->SqueezeDimZeroTimesTry(diff);
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn1;
  unsigned m_uBankInputTn2;
  unsigned m_uBankOutputTn;
};

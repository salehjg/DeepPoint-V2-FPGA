#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperTile: public CKernelWrapper{
 public:
  CKernelWrapperTile(
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

  CTensorBasePtr EnqueueKernelLaunch(unsigned parentLayerId, CTensorBasePtr inputTn, unsigned tileAxis, unsigned tileCount){
    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks
    unsigned rank = inputTn->GetRank();
    ConditionCheck(
        (rank==2 && tileAxis==1) ||
            (rank==2 && tileAxis==2) ||
            (rank==3 && tileAxis==2),
            "Unsupported input tensor rank and tileAxis combination."
    );

    if(rank==2){
      ConditionCheck(
          (rank==2 && tileAxis==1) || (rank==2 && tileAxis==2),
          "Unsupported input tensor rank and tileAxis combination."
      );
    }

    if(rank==3){
      ConditionCheck(rank==3 && tileAxis==2,"Unsupported input tensor rank and tileAxis combination.");
    }

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn = std::static_pointer_cast<CTensorXil<float>>(inputTn);
    auto xInputTn = pInputTn->CloneIfNeededToBank(m_uBankInputTn);
    if(m_bLogMemBankCrossings) m_vMemBankCrossings.push_back("abs("+ (pInputTn)->GetTensorTag() +"-tile_in)");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    unsigned _dim0=0,_dim1=0,_dim2=0;
    auto shape = xInputTn->GetShape();
    CTensorXilPtr<float> outputTn;

    if(rank==2){
      _dim0 = shape[0];
      _dim1 = shape[1];
      if(tileAxis==1){
        outputTn = CTensorXilPtr<float>(new CTensorXil<float>(GetXilInfo(), {_dim0,(unsigned int)tileCount,_dim1}, false, m_uBankOutputTn));
      }else if(tileAxis==2){
        outputTn = CTensorXilPtr<float>(new CTensorXil<float>(GetXilInfo(), {_dim0,_dim1,(unsigned int)tileCount}, false, m_uBankOutputTn));
      }else{
        // Something is not right.
        assert(false);
      }
    }else if(rank==3 && tileAxis==2){
      _dim0 = shape[0];
      _dim1 = shape[1];
      _dim2 = shape[2];
      outputTn = CTensorXilPtr<float>(new CTensorXil<float>(GetXilInfo(), {_dim0,_dim1,(unsigned)tileCount,_dim2}, false, m_uBankOutputTn));
    }else{
      // Something is not right.
      assert(false);
    }

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), _dim0));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), _dim1));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), _dim2));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), rank));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), tileAxis));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), tileCount));

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
    if(m_bLogMemBankCrossings) outputTn->SetTensorTag("tile_out");
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn;
  unsigned m_uBankOutputTn;
};

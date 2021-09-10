#pragma once

#include "fpga/xilinx/CKernelWrapper.h"
#include "CStringFormatter.h"
#include <iostream>
#include <vector>
#include <cassert>

class CKernelWrapperReduce: public CKernelWrapper{
 public:
  CKernelWrapperReduce(
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

  CTensorBasePtr EnqueueKernelLaunch(
      unsigned parentLayerId,
      CTensorBasePtr inputTn,
      REDUCTION_OPS mode,
      unsigned powY,
      const std::vector<unsigned> &combination){

    //-----------------------------------------------------------------------------------------------------------------
    // #. Requirement Checks
    unsigned rank = inputTn->GetRank();

    ConditionCheck(rank==combination.size(), "The combination's size must be equal to the input tensor's rank");

    if(mode == REDUCTION_OPS::SUM){
      ConditionCheck(rank==3 || rank==4, "For REDUCTION_OPS::SUM, only tensors of ranks 3 and 4 are supported.");
      if(rank==3){
        // ReduceSum3D FFT
        ConditionCheck(
             (!combination[0] && !combination[1] && combination[2]),
            "For REDUCTION_OPS::SUM and rank 3 tensor, only axis 2 reduction is supported."
        );
      }
      if(rank==4) {
        // ReduceSum4D TTTF
        ConditionCheck(
            (combination[0] && combination[1] && combination[2] && !combination[3]),
            "For REDUCTION_OPS::SUM and rank 4 tensors, only axes 0,1, and 2 reduction is supported."
        );
      }
    }else if (mode == REDUCTION_OPS::MAX){
      ConditionCheck(rank==4, "For REDUCTION_OPS::MAX, only rank 4 tensors are supported.");

      // ReduceMax4D TFTT && shape[2]==1
      // ReduceMax4D TTFT
      ConditionCheck(
          rank==4 && (
              ((!combination[0] && combination[1] && !combination[2] && !combination[3]) && inputTn->GetShape()[2]==1) ||
              (!combination[0] && !combination[1] && combination[2] && !combination[3])
          ),
          "For REDUCTION_OPS::MAX and rank 4 tensors, only FTFF with shape[2]=1 and FFTF combs are supported."
      );
    }

    // -----------------------------------------------------------------------------------------------------------------
    // #. Pointer Castings And Memory Bank Crossings
    auto pInputTn = std::static_pointer_cast<CTensorXil<float>>(inputTn);
    auto xInputTn = pInputTn->CloneIfNeededToBank(m_uBankInputTn);
    if(m_bLogMemBankCrossings) m_vMemBankCrossings.push_back("abs("+ (pInputTn)->GetTensorTag() +"-reduce_in)");

    // -----------------------------------------------------------------------------------------------------------------
    // #. Kernel Launch
    unsigned dim0=0, dim1=0, dim2=0, dim3=0;
    unsigned kernelMode=1000;
    auto shape = xInputTn->GetShape();
    CTensorXilPtr<float> outputTn;

    if(mode == REDUCTION_OPS::SUM){
      if(rank==3 && (!combination[0] && !combination[1] && combination[2])){
        // ReduceSum3D FFT
        dim0 = shape[0];
        dim1 = shape[1];
        dim2 = shape[2];
        kernelMode = 1;
        outputTn = CTensorXilPtr<float>(new CTensorXil<float>(
            GetXilInfo(),
            {shape[0],shape[1]},
            false,
            m_uBankOutputTn)
        );
      }else if(rank==4 && (combination[0] && combination[1] && combination[2] && !combination[3])){
        // ReduceSum4D TTTF
        dim0 = shape[0];
        dim1 = shape[1];
        dim2 = shape[2];
        dim3 = shape[3];
        kernelMode = 2;
        outputTn = CTensorXilPtr<float>(new CTensorXil<float>(
            GetXilInfo(),
            {shape[3]},
            false,
            m_uBankOutputTn)
        );
      }
    } else if (mode == REDUCTION_OPS::MAX){
      if(rank==4 && ((!combination[0] && combination[1] && !combination[2] && !combination[3]) && pInputTn->GetShape()[2]==1)){
        // ReduceMax4D FTFF && shape[2]==1
        // Uses the kernel for reduce max TFT
        dim0 = shape[0];
        dim1 = shape[1];
        dim2 = shape[3];
        outputTn = CTensorXilPtr<float>(new CTensorXil<float>(
            GetXilInfo(),
            {shape[0], 1, shape[3]},
            false,
            m_uBankOutputTn)
        );
        kernelMode = 3;

      }else if(rank==4 && (!combination[0] && !combination[1] && combination[2] && !combination[3])){
        // ReduceMax4D FFTF
        dim0 = shape[0]*shape[1];
        dim1 = shape[2];
        dim2 = shape[3];
        outputTn = CTensorXilPtr<float>(new CTensorXil<float>(
            GetXilInfo(),
            {shape[0], shape[1], shape[3]},
            false,
            m_uBankOutputTn)
        );
        kernelMode = 3;

      }else{
        assert(false); // something's wrong.
      }
    }else{
      assert(false); // something's wrong.
    }

    cl_int stat;
    ResetArgCounter();
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), xInputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), outputTn->GetDeviceBuffer()));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), kernelMode));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), powY));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim0));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim1));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim2));
    OclCheck(stat, stat = GetKernel()->setArg(ArgCounter(), dim3));

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
    if(m_bLogMemBankCrossings) outputTn->SetTensorTag("reduce_out");
    return std::dynamic_pointer_cast<CTensorBase>(outputTn);
  }

 private:
  unsigned m_uBankInputTn;
  unsigned m_uBankOutputTn;
};

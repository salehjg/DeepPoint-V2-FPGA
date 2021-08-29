#pragma once

#include "fpga/xilinx/xcl2.h"
#include "GlobalHelpers.h"
#include "CTensorBase.h"

class CXilinxInfo{
 public:
  CXilinxInfo(
      cl::Program *program,
      cl::Context *context,
      cl::CommandQueue *queue){
    m_oProgram = program;
    m_oContext = context;
    m_oQueue = queue;
    OclCheck(m_iStatus,
             m_oDatamoverKernel = new cl::Kernel(*program, "task_datamover", &m_iStatus)
    );
    m_oDummyDataMoverBank0 = nullptr;
    m_oDummyDataMoverBank1 = nullptr;
    m_oDummyDataMoverBank2 = nullptr;
    m_oDummyDataMoverBank3 = nullptr;
  }

  void SetDataMoverDummyTensors(
      CTensorBase *dummyTensorBank0,
      CTensorBase *dummyTensorBank1,
      CTensorBase *dummyTensorBank2,
      CTensorBase *dummyTensorBank3
      ){
#ifdef USEMEMORYBANK0
    ///TODO: MAXI_WIDTH SHOULD BE A PARAMETER. USING A GLOBAL DEF IS NOT A GOOD IDEA!
    ///      PUT SOME ASSERTS IN THE CLASSES THAT ARE GOING TO USE THESE TENSORS.
    m_oDummyDataMoverBank0 = dummyTensorBank0;
#endif
#ifdef USEMEMORYBANK1
    m_oDummyDataMoverBank1 = dummyTensorBank1;
#endif
#ifdef USEMEMORYBANK2
    m_oDummyDataMoverBank2 = dummyTensorBank2;
#endif
#ifdef USEMEMORYBANK3
    m_oDummyDataMoverBank3 = dummyTensorBank3;
#endif
  }

  cl::Program* GetProgram(){
    return m_oProgram;
  }

  cl::Context* GetContext(){
    return m_oContext;
  }

  cl::CommandQueue* GetQueue(){
    return m_oQueue;
  }

  cl::Kernel* GetDatamoverKernel(){
    return m_oDatamoverKernel;
  }

  CTensorBase* GetDatamoverDummyTensor(unsigned bankIndex){
    return (bankIndex==0)? m_oDummyDataMoverBank0 :
           (bankIndex==1)? m_oDummyDataMoverBank1 :
           (bankIndex==2)? m_oDummyDataMoverBank2 :
           (bankIndex==3)? m_oDummyDataMoverBank3 :
           nullptr;

  }

 protected:
 private:
  cl_int m_iStatus;
  cl::Program *m_oProgram;
  cl::Context *m_oContext;
  cl::CommandQueue *m_oQueue;
  cl::Kernel *m_oDatamoverKernel;
  CTensorBase *m_oDummyDataMoverBank0;
  CTensorBase *m_oDummyDataMoverBank1;
  CTensorBase *m_oDummyDataMoverBank2;
  CTensorBase *m_oDummyDataMoverBank3;
};
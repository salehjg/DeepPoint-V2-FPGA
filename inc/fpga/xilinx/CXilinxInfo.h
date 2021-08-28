#pragma once

#include "fpga/xilinx/xcl2.h"
#include "GlobalHelpers.h"
#include "fpga/xilinx/CTensorXil.h"

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

#ifdef USEMEMORYBANK0
    ///TODO: MAXI_WIDTH SHOULD BE A PARAMETER. USING A GLOBAL DEF IS NOT A GOOD IDEA!
    ///      PUT SOME ASSERTS IN THE CLASSES THAT ARE GOING TO USE THESE TENSORS.
    m_oDummyDataMoverBank0 = new CTensorXil<float>(context,queue,{50*1024*1024},true,0,CONFIG_M_AXI_WIDTH);
#endif
#ifdef USEMEMORYBANK1
    m_oDummyDataMoverBank1 = new CTensorXil<float>(context,queue,{50*1024*1024},true,1,CONFIG_M_AXI_WIDTH);
#endif
#ifdef USEMEMORYBANK2
    m_oDummyDataMoverBank2 = new CTensorXil<float>(context,queue,{50*1024*1024},true,2,CONFIG_M_AXI_WIDTH);
#endif
#ifdef USEMEMORYBANK3
    m_oDummyDataMoverBank3 = new CTensorXil<float>(context,queue,{50*1024*1024},true,3,CONFIG_M_AXI_WIDTH);
#endif

  }

  ~CXilinxInfo(){
#ifdef USEMEMORYBANK0
    delete m_oDummyDataMoverBank0
#endif
#ifdef USEMEMORYBANK1
    delete m_oDummyDataMoverBank1;
#endif
#ifdef USEMEMORYBANK2
    delete m_oDummyDataMoverBank2;
#endif
#ifdef USEMEMORYBANK3
    delete m_oDummyDataMoverBank3;
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

  CTensorXil<float>* GetDatamoverDummyTensor(unsigned bankIndex){
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
  CTensorXil<float> *m_oDummyDataMoverBank0;
  CTensorXil<float> *m_oDummyDataMoverBank1;
  CTensorXil<float> *m_oDummyDataMoverBank2;
  CTensorXil<float> *m_oDummyDataMoverBank3;
};
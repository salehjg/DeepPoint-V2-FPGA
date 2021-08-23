#include "fpga/xilinx/CKernelWrapper.h"

CKernelWrapper::CKernelWrapper(std::string &taskName,
                               std::string &fileName,
                               unsigned ddrBankIndex,
                               cl::Program *program,
                               std::string &path,
                               bool isDisabled) {
  m_strTaskName = taskName;
  m_strKernelName = fileName;
  m_strKernelPath = path;
  m_bIsDisabled = isDisabled;
  m_uBankIndex = ddrBankIndex;

  if(!m_bIsDisabled){
    OclCheck(m_iStatus,
        m_oKernel = new cl::Kernel(*program, m_strTaskName.c_str(), &m_iStatus));
  }
}

cl::Kernel *CKernelWrapper::GetKernel() const {
  return m_oKernel;
}

unsigned CKernelWrapper::GetBankIndex() const {
  return m_uBankIndex;
}

void CKernelWrapper::EventCallback(cl_event event, cl_int execStatus, void *userData) {
  if( execStatus != CL_COMPLETE ) {
    std::cout<<"ERROR IN KERNEL EXECUTION\n";
    ///TODO REPORT ERROR WITH SPDLOGGER
    return;
  }

  cl_ulong deviceTimeStart=0, deviceTimeEnd=0; //nano-seconds

  if(((CallbackData *) userData)->profileKernel){
    cl_int apiStatus;

    OclCheck(apiStatus,apiStatus = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START,&deviceTimeStart));
    OclCheck(apiStatus,apiStatus = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END,&deviceTimeEnd));

    cl_ulong durationNanoSeconds = deviceTimeEnd - deviceTimeStart;

    std::cout<<"KERNEL: ns:"<<durationNanoSeconds<<std::endl;
    ///TODO REPORT DURATION WITH SPDLOGGER
    /*SPDLOG_LOGGER_INFO(reporter,
                       "** {}{}(us): {}, (ms): {}, (s): {}",
                       m_strTaskName,
                       (false?"(ndrange):: ":"(task):: "),
                       durationNanoSeconds/1000.0f,
                       durationNanoSeconds/1000000.0f,
                       durationNanoSeconds/1000000000.0f);
    */
  }
}

CKernelWrapper& CKernelWrapper::AddArg(CKernelArgBase *arg) {
  m_vKernelArgs.push_back(arg);
  return this;
}
void CKernelWrapper::WipeKernelArgs() {
  for(auto a:m_vKernelArgs){
    delete a;
  }
  m_vKernelArgs.clear();
  m_vKernelArgsPrepared.clear();
}
void CKernelWrapper::PrepareInputTensors() {
  CKernelArg<CTensorXil<float>> *t1 = nullptr;
  CKernelArg<CTensorXil<unsigned>> *t2 = nullptr;

  // Look for CTensorXil arguments and clone them to the required mem bank (if needed) for the kernel launch.
  for(auto &arg : m_vKernelArgs){
    if(t1=dynamic_cast<CKernelArg<CTensorXil<float>>*>(arg)){
      auto argPrepared = t1->Get()->CloneIfNeededToBank(m_uBankIndex);
      m_vKernelArgsPrepared.push_back(argPrepared);
      m_vArgsToBeReleased.push_back(argPrepared);
    }else if(t2=dynamic_cast<CKernelArg<CTensorXil<unsigned>>*>(arg)){
      auto argPrepared = t2->Get()->CloneIfNeededToBank(m_uBankIndex);
      m_vKernelArgsPrepared.push_back(argPrepared);
      m_vArgsToBeReleased.push_back(argPrepared);
    }else{
      m_vKernelArgsPrepared.push_back(arg);
    }
  }
}

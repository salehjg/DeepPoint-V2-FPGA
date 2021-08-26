#include "fpga/xilinx/CKernelWrapper.h"

CKernelWrapper::CKernelWrapper(std::string &taskName,
                               std::string &fileName,
                               unsigned ddrBankIndex,
                               CXilinxInfo *xilInfo,
                               std::string &path,
                               bool isDisabled,
                               bool profileOcl) {
  m_iArgCounter = 0;
  m_strTaskName = taskName;
  m_strKernelName = fileName;
  m_strKernelPath = path;
  m_bIsDisabled = isDisabled;
  m_bProfileOcl = profileOcl;
  m_uBankIndex = ddrBankIndex;
  m_oXilInfo = xilInfo;

  if(!m_bIsDisabled){
    OclCheck(m_iStatus,
        m_oKernel = new cl::Kernel(*m_oXilInfo->GetProgram(), m_strTaskName.c_str(), &m_iStatus));
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
CXilinxInfo *CKernelWrapper::GetXilInfo() const {
  return m_oXilInfo;
}
void CKernelWrapper::ResetArgCounter() {
  m_iArgCounter = 0;
}
int CKernelWrapper::ArgCounter() {
  return m_iArgCounter++;
}
bool CKernelWrapper::GetProfileOclEnabled() const {
  return m_bProfileOcl;
}
bool CKernelWrapper::GetKernelEnabled() const {
  return m_bIsDisabled;
}

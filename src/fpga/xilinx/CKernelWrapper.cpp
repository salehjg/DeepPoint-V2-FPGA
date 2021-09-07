#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "fpga/xilinx/CKernelWrapper.h"

CKernelWrapper::CKernelWrapper(std::string taskName,
                               std::string fileName,
                               CXilinxInfo *xilInfo,
                               std::string path,
                               bool isEnabled,
                               bool profileOcl) {
  m_iArgCounter = 0;
  m_strTaskName = taskName;
  m_strKernelName = fileName;
  m_strKernelPath = path;
  m_bIsEnabled = isEnabled;
  m_bProfileOcl = profileOcl;
  m_oXilInfo = xilInfo;
  ResetBookKeeper();

  if(m_bIsEnabled){
    OclCheck(m_iStatus,
        m_oKernel = new cl::Kernel(*m_oXilInfo->GetProgram(), m_strTaskName.c_str(), &m_iStatus));
  }
}

cl::Kernel *CKernelWrapper::GetKernel() const {
  return m_oKernel;
}

void CKernelWrapper::EventCallback(cl_event event, cl_int execStatus, void *userData) {
  if( execStatus != CL_COMPLETE ) {
    std::cout<<"ERROR IN KERNEL EXECUTION\n";
    ///TODO REPORT ERROR WITH SPDLOGGER
    return;
  }

  cl_ulong deviceTimeStart=0, deviceTimeEnd=0; //nano-seconds

  if(((CallbackData *) userData)->profileKernel){
    cl_int stat;
    OclCheck(stat, stat=clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(deviceTimeStart), &deviceTimeStart, nullptr));
    OclCheck(stat, stat=clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(deviceTimeEnd), &deviceTimeEnd, nullptr));
    cl_ulong durationNanoSeconds = deviceTimeEnd - deviceTimeStart;
    std::cout<<"KERNEL: ns:"<<durationNanoSeconds<<std::endl;
    auto *classPtr = static_cast<CKernelWrapper*>(((CallbackData *)userData)->classPtr);
    classPtr->AddProfiledKernelLaunchDetails(classPtr->m_strTaskName, ((CallbackData *) userData)->parentLayerId, durationNanoSeconds);

    // Now that the async kernel is executed, we can release the smart pointers of the tensors required for this kernel.
    // Only the content of that row in the book-keeping vector is cleared; this is to make sure that the indexing system
    // would not get changed across multiple kernel launches.
    classPtr->ReleaseBookKeepingEntryAt( ((CallbackData *) userData)->kernelBookKeeperId);
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
bool CKernelWrapper::IsKernelEnabled() const {
  return m_bIsEnabled;
}
std::vector<ProfiledLaunchData> &CKernelWrapper::GetAccumulatedProfiledKernelLaunchData() {
  return m_vProfiledKernelLaunches;
}
void CKernelWrapper::AddProfiledKernelLaunchDetails(std::string taskName,
                                                    unsigned parentLayerId,
                                                    cl_ulong durationNanoSecOcl) {
  ProfiledLaunchData data;
  data.taskName = taskName;
  data.parentLayerId = parentLayerId;
  data.durationOcl = durationNanoSecOcl;
  m_vProfiledKernelLaunches.push_back(data);
}
void CKernelWrapper::ResetBookKeeper() {
  m_uBookKeeperCounter=0;
  m_vBookKeeper.clear();
}
unsigned CKernelWrapper::GetTotalTensorsInBookKeeper() {
  unsigned cnt=0;
  if(!m_vBookKeeper.empty()){
    for(auto &v:m_vBookKeeper){
      cnt += v.size();
    }
  }
  return cnt;
}
unsigned CKernelWrapper::GenerateBookKeeperId() {
  return m_uBookKeeperCounter++;
}
unsigned CKernelWrapper::GetTheLastBookKeeperId() {
  // returns -1 if empty otw a zero based index.
  return m_uBookKeeperCounter-1;
}
CallbackData* CKernelWrapper::GenerateAndStoreCallBackData(void* classPtr, unsigned parentLayerId) {
  CallbackData *obj = new CallbackData();
  obj->profileKernel = GetProfileOclEnabled();
  obj->kernelBookKeeperId = GenerateBookKeeperId();
  obj->classPtr = classPtr;
  obj->parentLayerId = parentLayerId;
  m_vCallBackData.push_back(obj);
  return obj;//m_vCallBackData.back();
}
void CKernelWrapper::StoreBookKeepingEntry(const std::vector<CTensorBasePtr> &vecTensorsToBePreserved) {
  m_vBookKeeper.push_back(vecTensorsToBePreserved);
}
void CKernelWrapper::ReleaseBookKeepingEntryAt(unsigned kernelBookKeepingId) {
  m_vBookKeeper.at(kernelBookKeepingId).clear();
}

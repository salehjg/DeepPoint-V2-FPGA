#include "CPlatformSelection.h"

CPlatformSelection::CPlatformSelection(bool loadWeights, std::string profilerOutputPath) {
  m_strProfilerOutputPath = profilerOutputPath;
  m_ptrProfiler = new CProfiler(m_strProfilerOutputPath);

  m_ptrImplCpu = new CImplementationCpu(m_ptrProfiler);
  m_ptrImplXil = new CImplementationXilinx(globalProfileOcl, m_ptrProfiler);
  m_ptrWeightsLoader = new CWeightLoader(m_ptrImplXil->GetXilInfo());


  if(!loadWeights) SPDLOG_LOGGER_WARN(logger,"The weights are not going to be loaded into the device memory.");
  if(loadWeights){
    std::string wDir = globalArgDataPath; wDir.append("/weights/");
    std::string wFileList = globalArgDataPath; wFileList.append("/weights/filelist.txt");
    SPDLOG_LOGGER_TRACE(logger,"Weights Dir: {}", wDir);
    SPDLOG_LOGGER_TRACE(logger,"Weights File List Path: {}", wFileList);
    m_ptrWeightsLoader->LoadWeightsFromDisk(wDir, wFileList);
  }
}

CPlatformSelection::~CPlatformSelection() {
  delete(m_ptrImplCpu);
  delete(m_ptrImplXil);
  delete(m_ptrWeightsLoader);
  delete(m_ptrProfiler);
}
template<typename T>
CTensorXil<T> *CPlatformSelection::CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensor<T> *srcTn) {
  assert(destPlatform==PLATFORMS::XIL);
  CTensorXil<T> *dstTn = new CTensorXil<T>(m_ptrImplXil->GetXilInfo(), *srcTn); // The default bank and AXI width.
  return dstTn;
}
template<typename T>
CTensor<T> *CPlatformSelection::CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorXil<T> *srcTn) {
  assert(destPlatform==PLATFORMS::CPU);
  CTensor<T> *dstTn = srcTn->TransferToHost();
  return dstTn;
}
CTensorBase *CPlatformSelection::CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorBase *srcTn) {
  using CpuFloat = CTensor<float>;
  using CpuUnsigned = CTensor<unsigned>;
  using XilFloat = CTensorXil<float>;
  using XilUnsigned = CTensorXil<unsigned>;

  if(srcTn->GetPlatform()==PLATFORMS::CPU){
    CpuFloat *cpuFloat;
    CpuUnsigned *cpuUnsigned;
    if(cpuFloat = dynamic_cast<CpuFloat*>(srcTn)){
      return CrossThePlatformIfNeeded<float>(destPlatform, cpuFloat);
    } else if(cpuUnsigned = dynamic_cast<CpuUnsigned*>(srcTn)){
      return CrossThePlatformIfNeeded<unsigned>(destPlatform, cpuUnsigned);
    }else{
      throw std::runtime_error(CStringFormatter() << __func__ << ": Undefined platform crossing type, please manually defined your used type.");
    }
  } else if(srcTn->GetPlatform()==PLATFORMS::XIL){
    XilFloat *xilFloat;
    XilUnsigned *xilUnsigned;
    if(xilFloat = dynamic_cast<XilFloat*>(srcTn)){
      return CrossThePlatformIfNeeded<float>(destPlatform, xilFloat);
    } else if(xilUnsigned = dynamic_cast<XilUnsigned*>(srcTn)){
      return CrossThePlatformIfNeeded<unsigned>(destPlatform, xilUnsigned);
    }else{
      throw std::runtime_error(CStringFormatter() << __func__ << ": Undefined platform crossing type, please manually defined your used type.");
    }
  }else{
    throw std::runtime_error(CStringFormatter() << __func__ << ": Undefined platform.");
  }

}
CTensorBase *CPlatformSelection::Concat2(PLATFORMS destPlatform, CTensorBase *inputTn1, CTensorBase *inputTn2, int concatAxis) {
  auto xinputTn1 = CrossThePlatformIfNeeded(destPlatform, inputTn1);
  auto xinputTn2 = CrossThePlatformIfNeeded(destPlatform, inputTn2);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Concat2(xinputTn1,xinputTn2,concatAxis);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Concat2(xinputTn1,xinputTn2,concatAxis);
  }else{
    throw std::runtime_error(CStringFormatter() << __func__ << ": Undefined platform.");
  }
}

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "CPlatformSelection.h"

CPlatformSelection::CPlatformSelection(bool loadWeights, bool oclProfiling, std::string profilerOutputPath) {
  m_bLoadWeights = loadWeights;
  m_bOclProfiling = oclProfiling;
  m_strProfilerOutputPath = profilerOutputPath;
  m_ptrProfiler = new CProfiler(m_strProfilerOutputPath);

  m_ptrImplCpu = new CImplementationCpu(m_ptrProfiler);
  m_ptrImplXil = new CImplementationXilinx(m_bOclProfiling, m_ptrProfiler);
  m_ptrWeightsLoader = new CWeightLoader(m_ptrImplXil->GetXilInfo());


  if(!m_bLoadWeights) SPDLOG_LOGGER_WARN(logger,"The weights are not going to be loaded into the device memory.");
  if(m_bLoadWeights){
    std::string wDir = globalArgDataPath; wDir.append("/weights/");
    std::string wFileList = globalArgDataPath; wFileList.append("/weights/filelist.txt");
    SPDLOG_LOGGER_TRACE(logger,"Weights Dir: {}", wDir);
    SPDLOG_LOGGER_TRACE(logger,"Weights File List Path: {}", wFileList);
    m_ptrWeightsLoader->LoadWeightsFromDisk(wDir, wFileList);
  }
}

CPlatformSelection::~CPlatformSelection() {
  SPDLOG_LOGGER_TRACE(logger, "Destroying CPlatformSelection().");
  delete(m_ptrImplCpu);
  SPDLOG_LOGGER_TRACE(logger, "CODE 1.");
  delete(m_ptrImplXil);
  SPDLOG_LOGGER_TRACE(logger, "CODE 2.");
  delete(m_ptrWeightsLoader);
  SPDLOG_LOGGER_TRACE(logger, "CODE 3.");
  delete(m_ptrProfiler);
  SPDLOG_LOGGER_TRACE(logger, "CODE 4.");
}

CTensorBase *CPlatformSelection::CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorBase *srcTn) {
  using CpuFloat = CTensor<float>;
  using CpuUnsigned = CTensor<unsigned>;
  using XilFloat = CTensorXil<float>;
  using XilUnsigned = CTensorXil<unsigned>;

  if(srcTn->GetPlatform()==PLATFORMS::CPU){
    if(destPlatform==PLATFORMS::XIL){
      CpuFloat *cpuFloat;
      CpuUnsigned *cpuUnsigned;
      if(cpuFloat = dynamic_cast<CpuFloat*>(srcTn)){
        return CrossThePlatform<float>(destPlatform, cpuFloat);
      } else if(cpuUnsigned = dynamic_cast<CpuUnsigned*>(srcTn)){
        return CrossThePlatform<unsigned>(destPlatform, cpuUnsigned);
      }else{
        ThrowException("Undefined platform crossing type, please manually defined your used type.");
      }
    }else{
      return srcTn;
    }
  } else if(srcTn->GetPlatform()==PLATFORMS::XIL){
    if(destPlatform==PLATFORMS::CPU){
      XilFloat *xilFloat;
      XilUnsigned *xilUnsigned;
      if(xilFloat = dynamic_cast<XilFloat*>(srcTn)){
        return CrossThePlatform<float>(destPlatform, xilFloat);
      } else if(xilUnsigned = dynamic_cast<XilUnsigned*>(srcTn)){
        return CrossThePlatform<unsigned>(destPlatform, xilUnsigned);
      }else{
        ThrowException("Undefined platform crossing type, please manually defined your used type.");
      }
    }else{
      return srcTn;
    }

  }else{
    ThrowException("Undefined platform.");
  }

}

CTensorBase *CPlatformSelection::Concat2(PLATFORMS destPlatform, CTensorBase *inputTn1, CTensorBase *inputTn2, int concatAxis) {
  if(!inputTn1->IsTypeFloat32() || !inputTn2->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto xinputTn1 = CrossThePlatformIfNeeded(destPlatform, inputTn1);
  auto xinputTn2 = CrossThePlatformIfNeeded(destPlatform, inputTn2);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Concat2(xinputTn1,xinputTn2,concatAxis);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Concat2(xinputTn1,xinputTn2,concatAxis);
  }else{
    ThrowException("Undefined Platform.");
  }
}


CImplementationXilinx *CPlatformSelection::GetClassPtrImplementationXilinx() {
  return m_ptrImplXil;
}

CProfiler *CPlatformSelection::GetClassPtrProfiler() {
  return m_ptrProfiler;
}

void CPlatformSelection::DumpToNumpyFile(PLATFORMS destPlatform,
                                         std::string npyFileName,
                                         CTensorBase *inputTn,
                                         std::string npyDumpDir) {
  auto xinputCpuTn = CrossThePlatformIfNeeded(PLATFORMS::CPU, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->DumpToNumpyFile(npyFileName,xinputCpuTn,npyDumpDir);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplCpu->DumpToNumpyFile(npyFileName,xinputCpuTn,npyDumpDir);
  }else{
    ThrowException("Undefined platform.");
  }
}

bool CPlatformSelection::CompareTensors(PLATFORMS destPlatform, CTensorBase *inputTn1, CTensorBase *inputTn2) {
  auto xinputCpuTn1 = CrossThePlatformIfNeeded(PLATFORMS::CPU, inputTn1);
  auto xinputCpuTn2 = CrossThePlatformIfNeeded(PLATFORMS::CPU, inputTn2);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->CompareTensors(xinputCpuTn1,xinputCpuTn2);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplCpu->CompareTensors(xinputCpuTn1,xinputCpuTn2);
  }else{
    ThrowException("Undefined platform.");
  }
}

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
  delete(m_ptrImplXil);
  delete(m_ptrWeightsLoader);
  delete(m_ptrProfiler);
}

CTensorBasePtr CPlatformSelection::CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorBasePtr srcTn) {
  using CpuFloat = CTensor<float>;
  using CpuUnsigned = CTensor<unsigned>;
  using XilFloat = CTensorXil<float>;
  using XilUnsigned = CTensorXil<unsigned>;

  if(srcTn->GetPlatform()==PLATFORMS::CPU){
    if(destPlatform==PLATFORMS::XIL){
      CTensorPtr<float> cpuFloat;
      CTensorPtr<unsigned> cpuUnsigned;
      if(cpuFloat = std::dynamic_pointer_cast<CpuFloat>(srcTn)){
        return CrossThePlatform<float>(destPlatform, cpuFloat);
      } else if(cpuUnsigned = std::dynamic_pointer_cast<CpuUnsigned>(srcTn)){
        return CrossThePlatform<unsigned>(destPlatform, cpuUnsigned);
      }else{
        ThrowException("Undefined platform crossing type, please manually defined your used type.");
      }
    }else{
      return srcTn;
    }
  } else if(srcTn->GetPlatform()==PLATFORMS::XIL){
    if(destPlatform==PLATFORMS::CPU){
      CTensorXilPtr<float> xilFloat;
      CTensorXilPtr<unsigned> xilUnsigned;
      if(xilFloat = std::dynamic_pointer_cast<XilFloat>(srcTn)){
        return CrossThePlatform<float>(destPlatform, xilFloat);
      } else if(xilUnsigned = std::dynamic_pointer_cast<XilUnsigned>(srcTn)){
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

CTensorBasePtr CPlatformSelection::Concat2(PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis) {
  if(!inputTn1->IsTypeFloat32() || !inputTn2->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn1 = CrossThePlatformIfNeeded(destPlatform, inputTn1);
  auto qInputTn2 = CrossThePlatformIfNeeded(destPlatform, inputTn2);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Concat2(qInputTn1,qInputTn2,concatAxis);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Concat2(qInputTn1,qInputTn2,concatAxis);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::MatMul(PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) {
  if(!inputTn1->IsTypeFloat32() || !inputTn2->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn1 = CrossThePlatformIfNeeded(destPlatform, inputTn1);
  auto qInputTn2 = CrossThePlatformIfNeeded(destPlatform, inputTn2);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->MatMul(qInputTn1,qInputTn2);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->MatMul(qInputTn1,qInputTn2);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::ReLU(PLATFORMS destPlatform, CTensorBasePtr inputTn) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->ReLU(qInputTn);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->ReLU(qInputTn);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::Sqrt(PLATFORMS destPlatform, CTensorBasePtr inputTn) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Sqrt(qInputTn);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Sqrt(qInputTn);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::Square(PLATFORMS destPlatform, CTensorBasePtr inputTn) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Square(qInputTn);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Square(qInputTn);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::BasicOps(PLATFORMS destPlatform,
                                            CTensorBasePtr inputTn1,
                                            CTensorBasePtr inputTn2,
                                            BASIC_OPS mode) {
  if(!inputTn1->IsTypeFloat32() || !inputTn2->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn1 = CrossThePlatformIfNeeded(destPlatform, inputTn1);
  auto qInputTn2 = CrossThePlatformIfNeeded(destPlatform, inputTn2);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->BasicOps(qInputTn1,qInputTn2,mode);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->BasicOps(qInputTn1,qInputTn2,mode);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::BasicOps(PLATFORMS destPlatform,
                                            CTensorBasePtr inputTn1,
                                            float scalar,
                                            BASIC_OPS mode) {
  if(!inputTn1->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn1 = CrossThePlatformIfNeeded(destPlatform, inputTn1);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->BasicOps(qInputTn1,scalar,mode);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->BasicOps(qInputTn1,scalar,mode);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::Tile(PLATFORMS destPlatform,
                                        CTensorBasePtr inputTn,
                                        unsigned tileAxis,
                                        unsigned tileCount) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Tile(qInputTn,tileAxis,tileCount);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Tile(qInputTn,tileAxis,tileCount);
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
                                         CTensorBasePtr inputTn,
                                         std::string npyDumpDir) {
  auto qInputCpuTn = CrossThePlatformIfNeeded(PLATFORMS::CPU, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    SPDLOG_LOGGER_TRACE(logger, "DumpToNumpyFile is not async meaning that it is blocking and will cause the ocl queue to be flushed.");
    return m_ptrImplCpu->DumpToNumpyFile(npyFileName,qInputCpuTn,npyDumpDir);
  }else if(destPlatform==PLATFORMS::XIL){
    ThrowException("NYI.");
  }else{
    ThrowException("Undefined platform.");
  }
}

bool CPlatformSelection::CompareTensors(PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) {
  auto qInputCpuTn1 = CrossThePlatformIfNeeded(PLATFORMS::CPU, inputTn1);
  auto qInputCpuTn2 = CrossThePlatformIfNeeded(PLATFORMS::CPU, inputTn2);
  if(destPlatform==PLATFORMS::CPU){
    SPDLOG_LOGGER_TRACE(logger, "CompareTensors is not async meaning that it is blocking and will cause the ocl queue to be flushed.");
    return m_ptrImplCpu->CompareTensors(qInputCpuTn1,qInputCpuTn2);
  }else if(destPlatform==PLATFORMS::XIL){
    ThrowException("NYI.");
  }else{
    ThrowException("Undefined platform.");
  }
}




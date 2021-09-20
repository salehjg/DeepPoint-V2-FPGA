#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "CPlatformSelection.h"

CPlatformSelection::CPlatformSelection(
    PLATFORMS targetPlatform,
    bool useShapeNetInstead,
    bool enableLoadingWeights,
    bool enableOclProfiling,
    bool enableMemBankCrossing,
    bool enableCpuUsageSampling,
    bool enableTensorDumps,
    std::string profilerOutputPath) {
  m_bUseShapeNet = useShapeNetInstead;
  m_bLoadWeights = enableLoadingWeights;
  m_bEnableOclProfiling = enableOclProfiling;
  m_bLogMemBankCrossings = enableMemBankCrossing;
  m_bEnableCpuUsageSampling = enableCpuUsageSampling;
  m_bEnableTensorDumps = enableTensorDumps;
  m_strProfilerOutputPath = profilerOutputPath;
  m_ptrProfiler = new CProfiler(m_strProfilerOutputPath, m_bEnableCpuUsageSampling);

  m_ptrImplCpu = new CImplementationCpu(m_ptrProfiler, m_bEnableTensorDumps);
  m_ptrImplXil = new CImplementationXilinx(m_ptrProfiler, m_bEnableOclProfiling, m_bLogMemBankCrossings);
  m_ptrWeightsLoader = new CWeightLoader(m_ptrImplXil->GetXilInfo(), targetPlatform);


  if(!m_bLoadWeights) SPDLOG_LOGGER_WARN(logger,"The weights are not going to be loaded into the device memory.");
  if(m_bLoadWeights){
    if(!m_bUseShapeNet){
      //ModelNet40
      std::string wDir = globalArgDataPath; wDir.append("/modelnet40/weights/");
      std::string wFileList = globalArgDataPath; wFileList.append("/modelnet40/weights/filelist.txt");
      SPDLOG_LOGGER_TRACE(logger,"Weights Dir: {}", wDir);
      SPDLOG_LOGGER_TRACE(logger,"Weights File List Path: {}", wFileList);
      m_ptrWeightsLoader->LoadWeightsFromDisk(wDir, wFileList);
    }else{
      //ShapeNet V2
      std::string wDir = globalArgDataPath; wDir.append("/shapenet2/weights/");
      std::string wFileList = globalArgDataPath; wFileList.append("/shapenet2/weights/filelist.txt");
      SPDLOG_LOGGER_TRACE(logger,"Weights Dir: {}", wDir);
      SPDLOG_LOGGER_TRACE(logger,"Weights File List Path: {}", wFileList);
      m_ptrWeightsLoader->LoadWeightsFromDisk(wDir, wFileList);
    }

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

CTensorBasePtr CPlatformSelection::Concat2(PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, unsigned concatAxis) {
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

CTensorBasePtr CPlatformSelection::Transpose(PLATFORMS destPlatform, CTensorBasePtr inputTn) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Transpose(qInputTn);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Transpose(qInputTn);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::Gather(PLATFORMS destPlatform,
                                          CTensorBasePtr inputTn,
                                          CTensorBasePtr indicesTn,
                                          unsigned indicesOfAxis) {
  if(!inputTn->IsTypeFloat32() || !indicesTn->IsTypeUint32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  auto qIndicesTn = CrossThePlatformIfNeeded(destPlatform, indicesTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Gather(qInputTn, qIndicesTn, indicesOfAxis);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Gather(qInputTn, qIndicesTn, indicesOfAxis);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::Reduce(PLATFORMS destPlatform,
                                          CTensorBasePtr inputTn,
                                          REDUCTION_OPS mode,
                                          unsigned powY,
                                          const std::vector<unsigned> &combination) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Reduce(qInputTn, mode, powY, combination);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Reduce(qInputTn, mode, powY, combination);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::Mean(PLATFORMS destPlatform,
                                        CTensorBasePtr inputTn,
                                        const std::vector<unsigned> &combination) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Mean(qInputTn, combination);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Mean(qInputTn, combination);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::Variance(PLATFORMS destPlatform,
                                            CTensorBasePtr inputTn,
                                            const std::vector<unsigned> &combination){
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Variance(qInputTn, combination);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Variance(qInputTn, combination);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::PadLastDim(PLATFORMS destPlatform, CTensorBasePtr inputTn, unsigned lastDimPadded) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->PadLastDim(qInputTn, lastDimPadded);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->PadLastDim(qInputTn, lastDimPadded);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::UnpadLastDim(PLATFORMS destPlatform,
                                                CTensorBasePtr inputTn,
                                                unsigned lastDimUnpadded) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->UnpadLastDim(qInputTn, lastDimUnpadded);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->UnpadLastDim(qInputTn, lastDimUnpadded);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::TopK(PLATFORMS destPlatform, CTensorBasePtr inputTn, unsigned axis, unsigned k) {
  if(!inputTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->TopK(qInputTn, axis, k);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->TopK(qInputTn, axis, k);
  }else{
    ThrowException("Undefined Platform.");
  }
}

CTensorBasePtr CPlatformSelection::Conv2D(PLATFORMS destPlatform,
                                          CTensorBasePtr inputTn,
                                          CTensorBasePtr weightTn,
                                          CTensorBasePtr biasTn) {
  if(!inputTn->IsTypeFloat32() || !weightTn->IsTypeFloat32() || !biasTn->IsTypeFloat32()){
    ThrowException("The layer only accepts types: float32.");
  }
  auto qInputTn = CrossThePlatformIfNeeded(destPlatform, inputTn);
  auto qWeightTn = CrossThePlatformIfNeeded(destPlatform, weightTn);
  auto qBiasTn = CrossThePlatformIfNeeded(destPlatform, biasTn);
  if(destPlatform==PLATFORMS::CPU){
    return m_ptrImplCpu->Conv2D(qInputTn,qWeightTn,qBiasTn);
  }else if(destPlatform==PLATFORMS::XIL){
    return m_ptrImplXil->Conv2D(qInputTn,qWeightTn,qBiasTn);
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

CWeightLoader *CPlatformSelection::GetClassPtrWeightLoader() {
  return m_ptrWeightsLoader;
}

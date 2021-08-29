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

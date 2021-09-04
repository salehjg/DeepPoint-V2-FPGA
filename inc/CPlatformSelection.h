#pragma once

#include "GlobalHelpers.h"
#include "CImplementationBase.h"
#include "cpu/CImplementationCpu.h"
#include "fpga/xilinx/CImplementationXilinx.h"
#include "CWeightLoader.h"
#include "CProfiler.h"
#include "GlobalHelpers.h"


class CPlatformSelection {
 public:
  CPlatformSelection(bool loadWeights, bool oclProfiling, std::string profilerOutputPath="profiler.json");
  ~CPlatformSelection();

  CTensorBasePtr Concat2(PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis);
  CTensorBasePtr MatMul(PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2);

  void DumpToNumpyFile(PLATFORMS platform, std::string npyFileName, CTensorBasePtr inputTn, std::string npyDumpDir=REPO_DIR"/data/matrix_dumps/");
  bool CompareTensors(PLATFORMS platform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2);

  CTensorBasePtr CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorBasePtr srcTn);
  CImplementationXilinx* GetClassPtrImplementationXilinx();
  CProfiler* GetClassPtrProfiler();

 private:
  template<typename T> CTensorXilPtr<T> CrossThePlatform(PLATFORMS destPlatform, CTensorPtr<T> srcTn);
  template<typename T> CTensorPtr<T> CrossThePlatform(PLATFORMS destPlatform, CTensorXilPtr<T> srcTn);

  CImplementationCpu *m_ptrImplCpu;
  CImplementationXilinx *m_ptrImplXil;
  CWeightLoader *m_ptrWeightsLoader;
  CProfiler *m_ptrProfiler;
  std::string m_strProfilerOutputPath;
  bool m_bOclProfiling, m_bLoadWeights;

};

// The template member functions of a non-template class should be declared and defined in the header file ONLY.
template<typename T>
CTensorXilPtr<T> CPlatformSelection::CrossThePlatform(PLATFORMS destPlatform, CTensorPtr<T> srcTn) {
  assert(destPlatform==PLATFORMS::XIL);
  auto *dstTn = new CTensorXil<T>(m_ptrImplXil->GetXilInfo(), *(srcTn.get())); // The default bank and AXI width.
  return CTensorXilPtr<T>(dstTn);
}
template<typename T>
CTensorPtr<T> CPlatformSelection::CrossThePlatform(PLATFORMS destPlatform, CTensorXilPtr<T> srcTn) {
  assert(destPlatform==PLATFORMS::CPU);
  return srcTn->TransferToHost();
}


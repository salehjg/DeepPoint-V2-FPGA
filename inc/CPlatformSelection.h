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

  CTensorBase *Concat2(PLATFORMS destPlatform, CTensorBase *inputTn1, CTensorBase *inputTn2, int concatAxis);
  void DumpToNumpyFile(PLATFORMS platform, std::string npyFileName, CTensorBase* inputTn, std::string npyDumpDir=REPO_DIR"/data/matrix_dumps/");
  bool CompareTensors(PLATFORMS platform, CTensorBase* inputTn1, CTensorBase* inputTn2);

  CTensorBase* CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorBase *srcTn);
  CImplementationXilinx* GetClassPtrImplementationXilinx();
  CProfiler* GetClassPtrProfiler();

 private:
  template<typename T> CTensorXil<T>* CrossThePlatform(PLATFORMS destPlatform, CTensor<T> *srcTn);
  template<typename T> CTensor<T>* CrossThePlatform(PLATFORMS destPlatform, CTensorXil<T> *srcTn);

  CImplementationCpu *m_ptrImplCpu;
  CImplementationXilinx *m_ptrImplXil;
  CWeightLoader *m_ptrWeightsLoader;
  CProfiler *m_ptrProfiler;
  std::string m_strProfilerOutputPath;
  bool m_bOclProfiling, m_bLoadWeights;

};

// The template member functions of a non-template class should be declared and defined in the header file ONLY.
template<typename T>
CTensorXil<T> *CPlatformSelection::CrossThePlatform(PLATFORMS destPlatform, CTensor<T> *srcTn) {
  assert(destPlatform==PLATFORMS::XIL);
  CTensorXil<T> *dstTn = new CTensorXil<T>(m_ptrImplXil->GetXilInfo(), *srcTn); // The default bank and AXI width.
  return dstTn;
}
template<typename T>
CTensor<T> *CPlatformSelection::CrossThePlatform(PLATFORMS destPlatform, CTensorXil<T> *srcTn) {
  assert(destPlatform==PLATFORMS::CPU);
  CTensor<T> *dstTn = srcTn->TransferToHost();
  return dstTn;
}


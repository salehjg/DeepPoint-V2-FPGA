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
  CPlatformSelection(bool loadWeights, std::string profilerOutputPath="profiler.json");
  CTensorBase *Concat2(PLATFORMS destPlatform, CTensorBase *inputTn1, CTensorBase *inputTn2, int concatAxis);

  CTensorBase* CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorBase *srcTn);
  template<typename T> CTensorXil<T>* CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensor<T> *srcTn);
  template<typename T> CTensor<T>* CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorXil<T> *srcTn);

  ~CPlatformSelection();
 private:
  CImplementationCpu *m_ptrImplCpu;
  CImplementationXilinx *m_ptrImplXil;
  CWeightLoader *m_ptrWeightsLoader;
  CProfiler *m_ptrProfiler;
  std::string m_strProfilerOutputPath;

};


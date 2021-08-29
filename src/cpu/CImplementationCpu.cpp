//
// Created by saleh on 8/26/21.
//

#include "cpu/CImplementationCpu.h"
CImplementationCpu::CImplementationCpu(CProfiler *profiler) {
  m_ePlatform = PLATFORMS::CPU;
  m_ptrProfiler = profiler;
  ResetLayerIdCounter(100000);
}
CTensorBase *CImplementationCpu::Concat2(CTensorBase *inputTn1, CTensorBase *inputTn2, int concatAxis) {
  return nullptr;
}

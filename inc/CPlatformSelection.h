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
  CPlatformSelection(
      PLATFORMS targetPlatform,
      bool useShapeNetInstead,
      bool enableLoadingWeights,
      bool enableOclProfiling,
      bool enableMemBankCrossing,
      bool enableCpuUtilization,
      bool enableTensorDumps,
      std::string profilerOutputPath="profiler.json");
  ~CPlatformSelection();

  CTensorBasePtr Concat2      (PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis);
  CTensorBasePtr MatMul       (PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2);
  CTensorBasePtr ReLU         (PLATFORMS destPlatform, CTensorBasePtr inputTn);
  CTensorBasePtr Sqrt         (PLATFORMS destPlatform, CTensorBasePtr inputTn);
  CTensorBasePtr Square       (PLATFORMS destPlatform, CTensorBasePtr inputTn);
  CTensorBasePtr BasicOps     (PLATFORMS destPlatform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, BASIC_OPS mode);
  CTensorBasePtr BasicOps     (PLATFORMS destPlatform, CTensorBasePtr inputTn1, float scalar, BASIC_OPS mode);
  CTensorBasePtr Tile         (PLATFORMS destPlatform, CTensorBasePtr inputTn, unsigned tileAxis, unsigned tileCount);
  CTensorBasePtr Transpose    (PLATFORMS destPlatform, CTensorBasePtr inputTn);
  CTensorBasePtr Gather       (PLATFORMS destPlatform, CTensorBasePtr inputTn, CTensorBasePtr indicesTn, unsigned indicesOfAxis);
  CTensorBasePtr Reduce       (PLATFORMS destPlatform, CTensorBasePtr inputTn, REDUCTION_OPS mode, unsigned powY, const std::vector<unsigned> &combination);
  CTensorBasePtr Mean         (PLATFORMS destPlatform, CTensorBasePtr inputTn, const std::vector<unsigned> &combination);
  CTensorBasePtr Variance     (PLATFORMS destPlatform, CTensorBasePtr inputTn, const std::vector<unsigned> &combination);
  CTensorBasePtr PadLastDim   (PLATFORMS destPlatform, CTensorBasePtr inputTn, unsigned lastDimPadded);
  CTensorBasePtr UnpadLastDim (PLATFORMS destPlatform, CTensorBasePtr inputTn, unsigned lastDimUnpadded);
  CTensorBasePtr TopK         (PLATFORMS destPlatform, CTensorBasePtr inputTn, unsigned axis, unsigned k);
  CTensorBasePtr Conv2D       (PLATFORMS destPlatform, CTensorBasePtr inputTn, CTensorBasePtr weightTn, CTensorBasePtr biasTn);

  void DumpToNumpyFile(PLATFORMS platform, std::string npyFileName, CTensorBasePtr inputTn, std::string npyDumpDir=REPO_DIR"/data/matrix_dumps/");
  bool CompareTensors(PLATFORMS platform, CTensorBasePtr inputTn1, CTensorBasePtr inputTn2);

  CTensorBasePtr CrossThePlatformIfNeeded(PLATFORMS destPlatform, CTensorBasePtr srcTn);
  CImplementationXilinx* GetClassPtrImplementationXilinx();
  CProfiler* GetClassPtrProfiler();
  CWeightLoader* GetClassPtrWeightLoader();

 private:
  template<typename T> CTensorXilPtr<T> CrossThePlatform(PLATFORMS destPlatform, CTensorPtr<T> srcTn);
  template<typename T> CTensorPtr<T> CrossThePlatform(PLATFORMS destPlatform, CTensorXilPtr<T> srcTn);

  CImplementationCpu *m_ptrImplCpu;
  CImplementationXilinx *m_ptrImplXil;
  CWeightLoader *m_ptrWeightsLoader;
  CProfiler *m_ptrProfiler;
  std::string m_strProfilerOutputPath;
  bool m_bEnableOclProfiling, m_bUseShapeNet, m_bLoadWeights, m_bLogMemBankCrossings, m_bEnableCpuUsageSampling, m_bEnableTensorDumps;

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

#pragma once

#include <string>
#include "CImplementationBase.h"
#include "fpga/xilinx/xcl2.h"
#include "CProfiler.h"
#include "CXilinxInfo.h"
#include "fpga/xilinx/kernels/CKernelWrapperConcat.h"

enum class RUN_MODE{
  SwEmu,
  HwEmu,
  Hw,
  Unknown
};

#define KERNEL_ENABLED true
#define KERNEL_DISABLED true

class CImplementationXilinx: public CImplementationBase {
 public:

  CImplementationXilinx(bool profileOcl, CProfiler *profiler);
  ~CImplementationXilinx();
  CXilinxInfo* GetXilInfo();
  int SetModeEnvVar(RUN_MODE &mode);
  RUN_MODE GetModeEnvVar() const;
  const std::string GetOclErrorMessage(cl_int error) const;

  CTensorBase* Concat2(CTensorBase* inputTn1, CTensorBase* inputTn2, int concatAxis) override;

 private:
  bool m_bOclProfileEnabled;
  CXilinxInfo *m_ptrXilInfo;
  cl_int m_iStatus;
  const std::string KERNEL_DIR = REPO_DIR "src/fpga/xilinx/kernels";
  std::string m_strDeviceName;
  cl::Device m_oDevice;
  cl::Context *m_ptrContext;
  cl::Program *m_ptrProgram;
  cl::CommandQueue *m_ptrQueue;
  CTensorXil<float> *m_ptrDataMoverDummyTensorBank0;
  CTensorXil<float> *m_ptrDataMoverDummyTensorBank1;
  CTensorXil<float> *m_ptrDataMoverDummyTensorBank2;
  CTensorXil<float> *m_ptrDataMoverDummyTensorBank3;
  std::vector<ProfiledLaunchData> *m_ptrDataMoverProfiledDataVec;

  CKernelWrapperConcat *m_ptrKernelConcat;
};



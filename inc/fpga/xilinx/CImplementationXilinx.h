#pragma once

#include <string>
#include "CImplementationBase.h"
#include "fpga/xilinx/xcl2.h"
#include "CProfiler.h"
#include "CXilinxInfo.h"

enum class RUN_MODE{
  SwEmu,
  HwEmu,
  Hw,
  Unknown
};

class CImplementationXilinx: public CImplementationBase {
 public:
  CImplementationXilinx();

  int SetModeEnvVar(RUN_MODE &mode);
  RUN_MODE GetModeEnvVar() const;
  const std::string& GetOclErrorMessage(cl_int error) const;

 private:
  CXilinxInfo *m_ptrXilInfo;
  cl_int m_iStatus;
  const std::string KERNEL_DIR = REPO_DIR "src/fpga/xilinx/kernels";
  std::string m_strDeviceName;
  cl::Device m_oDevice;
  cl::Context *m_ptrContext;
  cl::Program *m_ptrProgram;
  cl::CommandQueue *m_ptrQueue;
};



#pragma once

#include <string>
#include <algorithm>
#include "fpga/xilinx/xcl2.h"
#include "fpga/xilinx/CTensorXil.h"
#include "CTensorBase.h"
#include "GlobalHelpers.h"
#include "fpga/xilinx/CKernelArg.h"


struct CallbackData{
  int execId;
  bool profileKernel;
};

class CKernelWrapper {
 public:
  CKernelWrapper(
      std::string &taskName,
      std::string &fileName,
      unsigned ddrBankIndex,
      cl::Program *program,
      std::string &path,
      bool isDisabled);

  cl::Kernel* GetKernel() const;
  unsigned GetBankIndex() const;
  void WipeKernelArgs();
  CKernelWrapper& AddArg(CKernelArgBase *arg);
  void PrepareInputTensors();

 protected:
  void EventCallback(cl_event event, cl_int execStatus, void* userData);
 private:
  std::string m_strTaskName, m_strKernelName, m_strKernelPath;
  unsigned m_uBankIndex;
  cl::Kernel *m_oKernel;
  bool m_bIsDisabled;
  cl_int m_iStatus;
  std::vector<CKernelArgBase*> m_vKernelArgs;
  std::vector<CKernelArgBase*> m_vKernelArgsPrepared;

  // these should be destroyed after all of the async kernels have been executed.
  std::vector<CKernelArgBase*> m_vArgsToBeReleased;
};


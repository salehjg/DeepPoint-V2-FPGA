#pragma once

#include <string>
#include <algorithm>
#include "fpga/xilinx/xcl2.h"
#include "fpga/xilinx/CTensorXil.h"
#include "CTensorBase.h"
#include "GlobalHelpers.h"
#include "fpga/xilinx/CXilinxInfo.h"
#include <cassert>


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
      CXilinxInfo *xilInfo,
      std::string &path,
      bool isDisabled);

  cl::Kernel* GetKernel() const;
  unsigned GetBankIndex() const;
  CXilinxInfo* GetXilInfo() const;
  void ResetArgCounter();
  int ArgCounter();
 protected:
  void EventCallback(cl_event event, cl_int execStatus, void* userData);
 private:
  std::string m_strTaskName, m_strKernelName, m_strKernelPath;
  unsigned m_uBankIndex;
  cl::Kernel *m_oKernel;
  bool m_bIsDisabled;
  cl_int m_iStatus;
  CXilinxInfo *m_oXilInfo;
  int m_iArgCounter;
};


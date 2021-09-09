#pragma once

#include <string>
#include <algorithm>
#include "fpga/xilinx/xcl2.h"
#include "fpga/xilinx/CTensorXil.h"
#include "CTensorBase.h"
#include "GlobalHelpers.h"
#include "fpga/xilinx/CXilinxInfo.h"
#include <cassert>
#include "GlobalHelpers.h"

class CKernelWrapper {
 public:
  CKernelWrapper(
      std::string taskName,
      std::string fileName,
      CXilinxInfo *xilInfo,
      std::string path,
      bool isDisabled,
      bool profileOcl,
      bool logMemBankCrossings);
  virtual ~CKernelWrapper();

  cl::Kernel* GetKernel() const;
  CXilinxInfo* GetXilInfo() const;
  void ResetArgCounter();
  int ArgCounter();
  bool GetProfileOclEnabled() const;
  bool IsKernelEnabled() const;
  std::vector<ProfiledLaunchData>& GetAccumulatedProfiledKernelLaunchData();
  void ResetBookKeeper();
  unsigned GetTotalTensorsInBookKeeper();
  unsigned GenerateBookKeeperId();
  unsigned GetTheLastBookKeeperId();
  CallbackData* GenerateAndStoreCallBackData(void* classPtr, unsigned parentLayerId);
  void StoreBookKeepingEntry(const std::vector<CTensorBasePtr> &vecTensorsToBePreserved);
  void ReleaseBookKeepingEntryAt(unsigned kernelBookKeepingId);

 protected:
  static void EventCallback(cl_event event, cl_int execStatus, void* userData);
  void AddProfiledKernelLaunchDetails(std::string taskName, unsigned parentLayerId, cl_ulong durationNanoSecOcl);

  std::vector<CallbackData*> m_vCallBackData;
  std::vector<std::vector<CTensorBasePtr>> m_vBookKeeper;
  std::vector<std::string> m_vMemBankCrossings;
  bool m_bLogMemBankCrossings;

 private:
  std::string m_strTaskName, m_strKernelName, m_strKernelPath;
  cl::Kernel *m_oKernel;
  bool m_bIsEnabled;
  bool m_bProfileOcl;
  cl_int m_iStatus;
  CXilinxInfo *m_oXilInfo;
  int m_iArgCounter;
  std::vector<ProfiledLaunchData> m_vProfiledKernelLaunches;
  unsigned m_uBookKeeperCounter;
};


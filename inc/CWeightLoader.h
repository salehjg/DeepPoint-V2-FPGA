#pragma once
#include <fstream>
#include <algorithm>
#include <string>
#include <cassert>
#include <vector>
#include "GlobalHelpers.h"
#include "CTensorBase.h"
#include "cpu/CTensor.h"
#include "fpga/xilinx/CTensorXil.h"
#include "cnpy.h"
#include "fpga/xilinx/CXilinxInfo.h"

class CWeightLoader {
 public:
  CWeightLoader(CXilinxInfo *xilInfo);
  ~CWeightLoader();
  void LoadWeightsFromDisk(
      std::string &weightsBaseDir,
      std::string &pathToTxtFnameList);
  CTensorBasePtr AccessWeights(PLATFORMS platform, std::string &name);

 private:
  int ResolveMemoryBank(PLATFORMS platform, std::string &name);
  int _ResolveMemoryBankOclXilinx(std::string &name);
  std::string _ResolveTensorTagOclXilinx(std::string &name);

  bool m_bIsLoaded;
  CXilinxInfo *m_ptrXilInfo;
  unsigned m_uWeightCount;
  std::vector<CTensorBasePtr> m_vWeightsCpu;
  std::vector<CTensorBasePtr> m_vWeightsXil;
  std::map<std::string,int> m_mWeightNameToIndex;
  std::vector<cnpy::NpyArray> m_vNumpyBuff;
};


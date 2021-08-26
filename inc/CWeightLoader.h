#pragma once

#include <vector>
#include "GlobalHelpers.h"
#include "CTensorBase.h"
#include "cpu/CTensor.h"
#include "fpga/xilinx/CTensorXil.h"
#include "cnpy.h"
#include "fpga/xilinx/CXilinxInfo.h"

class CWeightLoader {
 public:
  CWeightLoader();
  void LoadWeightsFromDisk(
      std::string &weightsBaseDir,
      std::string &pathToTxtFnameList,
      CXilinxInfo *xilInfo);
  CTensorBase* AccessWeights(PLATFORMS &platform, std::string &name);

 private:
  int ResolveMemoryBank(PLATFORMS &platform, std::string &name);
  int _ResolveMemoryBankOclXilinx(std::string &name);
  std::string _ResolveTensorTagOclXilinx(std::string &name);

  std::map<std::string,int> strToIndexMap;
  std::vector<cnpy::NpyArray> _cnpyBuff;
  CTensor<float>** weightsCPU;
  CTensorXil<float>** weightsXil;
};


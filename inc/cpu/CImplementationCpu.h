#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include "xilinx/config.h"
#include "CImplementationBase.h"
#include "CTensor.h"
#include "CProfiler.h"
#include "cnpy.h"

class CImplementationCpu: public CImplementationBase {
 public:
  CImplementationCpu(CProfiler *profiler);

  CTensorBasePtr Concat2(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis) override;
  CTensorBasePtr MatMul(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) override;
  //CTensorBase* Transpose(CTensorBase *inputTn);
  //;
  ////CTensorBase* MatMul(CTensorBase* inputTn, float scalar);
  //CTensorBase* Square(CTensorBase* inputTn);
  //CTensorBase* ReduceSum(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2);
  //CTensorBase* ReduceSum4D(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3);
  //CTensorBase* Mean(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3);
  //CTensorBase* Variance(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3);
  //CTensorBase* MatOps(CTensorBase *inputTn1, CTensorBase *inputTn2, MAT_OPS mode);
  //CTensorBase* MatOps(CTensorBase *inputTn1, float scalar, MAT_OPS mode);
  //CTensorBase* Sqrt(CTensorBase* inputTn);
  //CTensorBase* ReduceMax(CTensorBase* inputTn, int reductionDim);
  //CTensorBase* TopK(CTensorBase* inputTn, int axis, int k);
  //CTensorBase* Gather(CTensorBase* inputTn, CTensorBase* indicesTn, int indicesOfAxis);
  //CTensorBase* Conv2D(CTensorBase* inputTn, CTensorBase* weightsTn, CTensorBase* biasesTn);
  //CTensorBase* ReLU(CTensorBase* inputTn);
  //CTensorBase* Tile(CTensorBase *inputTn, int tileAxis, int tileCount);
  //CTensorBase* PadLastDim(CTensorBase* inputTn, unsigned lastDimPadded);
  //CTensorBase* UnpadLastDim(CTensorBase* inputTn, unsigned lastDimUnpadded);

  void DumpToNumpyFile(std::string npyFileName, CTensorBasePtr inputTn, std::string npyDumpDir);
  bool CompareTensors(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2);

 private:
  template <typename T> void DumpToNumpyFile(std::string npyFileName, CTensorPtr<T> inputTn, std::string npyDumpDir);
  template <typename T> bool CompareTensors(CTensorPtr<T> inputTn1, CTensorPtr<T> inputTn2);
};

template<typename T>
void CImplementationCpu::DumpToNumpyFile(std::string npyFileName, CTensorPtr<T> inputTn, std::string npyDumpDir) {
  auto shape = inputTn->GetShape();
  std::vector<unsigned long> _shape(shape.begin(), shape.end());
  cnpy::npy_save<T>(npyDumpDir+npyFileName, inputTn->Get(), _shape, "w");
}

template<typename T>
bool CImplementationCpu::CompareTensors(CTensorPtr<T> inputTn1, CTensorPtr<T> inputTn2) {
  if(inputTn1->GetShape()!=inputTn2->GetShape()) return false;
  bool matches = true;
  for(size_t i=0; i<inputTn1->GetLen(); i++){
    if((*inputTn1)[i]!=(*inputTn2)[i]){
      SPDLOG_LOGGER_ERROR(
          logger,
          "CompareTensors: Mismatch at tn1[{}]={}, tn2[{}]={}",
          i,
          (*inputTn1)[i],
          i,
          (*inputTn2)[i]
      );
      matches = false;
    }
  }
  return matches;
}

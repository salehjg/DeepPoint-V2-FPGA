#pragma once

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

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

  CTensorBasePtr Concat2      (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis) override;
  CTensorBasePtr MatMul       (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) override;
  CTensorBasePtr ReLU         (CTensorBasePtr inputTn) override;
  CTensorBasePtr Sqrt         (CTensorBasePtr inputTn) override;
  CTensorBasePtr Square       (CTensorBasePtr inputTn) override;
  CTensorBasePtr BasicOps     (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, BASIC_OPS mode) override;
  CTensorBasePtr BasicOps     (CTensorBasePtr inputTn1, float scalar, BASIC_OPS mode) override;
  CTensorBasePtr Tile         (CTensorBasePtr inputTn, unsigned tileAxis, unsigned tileCount) override;
  CTensorBasePtr Transpose    (CTensorBasePtr inputTn) override;
  CTensorBasePtr Gather       (CTensorBasePtr inputTn, CTensorBasePtr indicesTn, unsigned indicesOfAxis) override;
  CTensorBasePtr Reduce       (CTensorBasePtr inputTn, REDUCTION_OPS mode, unsigned powY, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3) override ;
  CTensorBasePtr Mean         (CTensorBasePtr inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3) override;
  CTensorBasePtr Variance     (CTensorBasePtr inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3) override;
  CTensorBasePtr PadLastDim   (CTensorBasePtr inputTn, unsigned lastDimPadded) override;
  CTensorBasePtr UnpadLastDim (CTensorBasePtr inputTn, unsigned lastDimUnpadded) override;
  CTensorBasePtr TopK         (CTensorBasePtr inputTn, unsigned axis, unsigned k) override;
  CTensorBasePtr Conv2D       (CTensorBasePtr inputTn, CTensorBasePtr weightTn, CTensorBasePtr biasTn) override;



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
  const float tolerance = 0.005f;
  if(inputTn1->GetShape()!=inputTn2->GetShape()) return false;
  bool matches = true;
  for(size_t i=0; i<inputTn1->GetLen(); i++){
    if(fabsf( (*inputTn1)[i]-(*inputTn2)[i] ) > tolerance){
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

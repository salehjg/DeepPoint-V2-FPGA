#pragma once
#include "CTensorBase.h"
#include "GlobalHelpers.h"
#include "CStringFormatter.h"
#include "CProfiler.h"
#include <string>

class CImplementationBase {
 public:
  //virtual CTensorBase* Transpose(CTensorBase *inputTn)=0;
  //virtual CTensorBase* MatMul(CTensorBase* inputTn1, CTensorBase* inputTn2)=0;
  ////virtual CTensorBase* MatMul(CTensorBase* inputTn, float scalar)=0;
  //virtual CTensorBase* Square(CTensorBase* inputTn)=0;
  //virtual CTensorBase* ReduceSum(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2)=0;
  //virtual CTensorBase* ReduceSum4D(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3)=0;
  //virtual CTensorBase* Mean(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3)=0;
  //virtual CTensorBase* Variance(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3)=0;
  //virtual CTensorBase* MatOps(CTensorBase *inputTn1, CTensorBase *inputTn2, MAT_OPS mode)=0;
  //virtual CTensorBase* MatOps(CTensorBase *inputTn1, float scalar, MAT_OPS mode)=0;
  //virtual CTensorBase* Sqrt(CTensorBase* inputTn)=0;
  virtual CTensorBase* Concat2(CTensorBase* inputTn1, CTensorBase* inputTn2, int concatAxis)=0;
  //virtual CTensorBase* ReduceMax(CTensorBase* inputTn, int reductionDim)=0;
  //virtual CTensorBase* TopK(CTensorBase* inputTn, int axis, int k)=0;
  //virtual CTensorBase* Gather(CTensorBase* inputTn, CTensorBase* indicesTn, int indicesOfAxis)=0;
  //virtual CTensorBase* Conv2D(CTensorBase* inputTn, CTensorBase* weightsTn, CTensorBase* biasesTn)=0;
  //virtual CTensorBase* ReLU(CTensorBase* inputTn)=0;
  //virtual CTensorBase* Tile(CTensorBase *inputTn, int tileAxis, int tileCount)=0;
  //virtual void         DumpMatrix(std::string npyFilename, CTensorBase* inputTn, std::string npyDir)=0;
  //virtual bool         CompareTensors(CTensorBase* inputTn1, CTensorBase* inputTn2)=0;
  //virtual bool         CompareTensorsInteger(CTensorBase* inputTn1, CTensorBase* inputTn2)=0;
  //virtual CTensorBase* PadLastDim(CTensorBase* inputTn, unsigned lastDimPadded)=0;
  //virtual CTensorBase* UnpadLastDim(CTensorBase* inputTn, unsigned lastDimUnpadded)=0;

 protected:
  unsigned GenerateLayerId();
  unsigned GetTheLastLayerId();
  PLATFORMS GetPlatform() const;
  void ResetLayerIdCounter(unsigned offset);
  void ValidateTensorPlatforms(const std::vector<CTensorBase*> &tensors, PLATFORMS requiredPlatform);

  std::atomic_uint m_uAtomicCounter;
  PLATFORMS m_ePlatform;
  CProfiler *m_ptrProfiler;
};

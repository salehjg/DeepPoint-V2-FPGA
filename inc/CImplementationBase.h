#pragma once
#include "CTensorBase.h"
#include "GlobalHelpers.h"
#include "CStringFormatter.h"
#include "CProfiler.h"
#include <string>

class CImplementationBase {
 public:
  virtual CTensorBasePtr Concat2      (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis)=0;
  virtual CTensorBasePtr MatMul       (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2)=0;
  virtual CTensorBasePtr ReLU         (CTensorBasePtr inputTn)=0;
  virtual CTensorBasePtr Sqrt         (CTensorBasePtr inputTn)=0;
  virtual CTensorBasePtr Square       (CTensorBasePtr inputTn)=0;
  virtual CTensorBasePtr BasicOps     (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, BASIC_OPS mode)=0;
  virtual CTensorBasePtr BasicOps     (CTensorBasePtr inputTn1, float scalar, BASIC_OPS mode)=0;
  virtual CTensorBasePtr Tile         (CTensorBasePtr inputTn, unsigned tileAxis, unsigned tileCount)=0;
  virtual CTensorBasePtr Transpose    (CTensorBasePtr inputTn)=0;
  virtual CTensorBasePtr Gather       (CTensorBasePtr inputTn, CTensorBasePtr indicesTn, unsigned indicesOfAxis)=0;
  virtual CTensorBasePtr Reduce       (CTensorBasePtr inputTn, REDUCTION_OPS mode, unsigned powY, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3)=0;
  virtual CTensorBasePtr Mean         (CTensorBasePtr inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3)=0;
  virtual CTensorBasePtr Variance     (CTensorBasePtr inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3)=0;
  virtual CTensorBasePtr PadLastDim   (CTensorBasePtr inputTn, unsigned lastDimPadded)=0;
  virtual CTensorBasePtr UnpadLastDim (CTensorBasePtr inputTn, unsigned lastDimUnpadded)=0;
  virtual CTensorBasePtr TopK         (CTensorBasePtr inputTn, unsigned axis, unsigned k)=0;
  virtual CTensorBasePtr Conv2D       (CTensorBasePtr inputTn, CTensorBasePtr weightTn, CTensorBasePtr biasTn)=0;

  //virtual void         DumpMatrix(std::string npyFilename, CTensorBase* inputTn, std::string npyDir)=0;
  //virtual bool         CompareTensors(CTensorBase* inputTn1, CTensorBase* inputTn2)=0;


 protected:
  unsigned GenerateLayerId();
  unsigned GetTheLastLayerId();
  PLATFORMS GetPlatform() const;
  void ResetLayerIdCounter(unsigned offset);
  void ValidateTensorPlatforms(const std::vector<CTensorBasePtr> &tensors, PLATFORMS requiredPlatform);

  std::atomic_uint m_uAtomicCounter;
  PLATFORMS m_ePlatform;
  CProfiler *m_ptrProfiler;
};

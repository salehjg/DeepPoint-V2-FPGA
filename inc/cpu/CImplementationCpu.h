#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include "xilinx/config.h"
#include "CImplementationBase.h"
#include "CTensor.h"
#include "CProfiler.h"

class CImplementationCpu: public CImplementationBase {
  CImplementationCpu(CProfiler *profiler);

  CTensorBase* Transpose(CTensorBase *inputTn);
  CTensorBase* MatMul(CTensorBase* inputTn1, CTensorBase* inputTn2);
  //CTensorBase* MatMul(CTensorBase* inputTn, float scalar);
  CTensorBase* Square(CTensorBase* inputTn);
  CTensorBase* ReduceSum(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2);
  CTensorBase* ReduceSum4D(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3);
  CTensorBase* Mean(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3);
  CTensorBase* Variance(CTensorBase* inputTn, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3);
  CTensorBase* MatOps(CTensorBase *inputTn1, CTensorBase *inputTn2, MAT_OPS mode);
  CTensorBase* MatOps(CTensorBase *inputTn1, float scalar, MAT_OPS mode);
  CTensorBase* Sqrt(CTensorBase* inputTn);
  CTensorBase* Concat2(CTensorBase* inputTn1, CTensorBase* inputTn2, int concatAxis);
  CTensorBase* ReduceMax(CTensorBase* inputTn, int reductionDim);
  CTensorBase* TopK(CTensorBase* inputTn, int axis, int k);
  CTensorBase* Gather(CTensorBase* inputTn, CTensorBase* indicesTn, int indicesOfAxis);
  CTensorBase* Conv2D(CTensorBase* inputTn, CTensorBase* weightsTn, CTensorBase* biasesTn);
  CTensorBase* ReLU(CTensorBase* inputTn);
  CTensorBase* Tile(CTensorBase *inputTn, int tileAxis, int tileCount);
  void         DumpMatrix(std::string npyFilename, CTensorBase* inputTn, std::string npyDir);
  bool         CompareTensors(CTensorBase* inputTn1, CTensorBase* inputTn2);
  bool         CompareTensorsInteger(CTensorBase* inputTn1, CTensorBase* inputTn2);
  CTensorBase* PadLastDim(CTensorBase* inputTn, unsigned lastDimPadded);
  CTensorBase* UnpadLastDim(CTensorBase* inputTn, unsigned lastDimUnpadded);

};

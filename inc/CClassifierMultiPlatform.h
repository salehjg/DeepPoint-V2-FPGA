#pragma once

#include "models/CModel1.h"
#include "cpu/CTensor.h"

class CClassifierMultiPlatform {
 public:
  CClassifierMultiPlatform();
  double GetTimestamp();
  void CalculateAccuracy(CTensor<float>* scores, CTensor<int>* labels, unsigned batchSize, unsigned classCount);

 private:
  CModel1 *m_ptrClassifierModel;
};

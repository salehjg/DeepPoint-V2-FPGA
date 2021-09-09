#pragma once

#include "models/CModel1.h"
#include "cpu/CTensor.h"

class CClassifierMultiPlatform {
 public:
  CClassifierMultiPlatform(
      bool useShapeNetInstead,
      bool enableOclProfiling,
      bool enableMemBankCrossing,
      bool enableCpuUtilization,
      bool enableTensorDumps);
  double GetTimestamp();

 private:
  void CalculateAccuracy(CTensorPtr<float> scores, CTensorPtr<unsigned> labels, unsigned batchSize, unsigned classCount);

  CModel1 *m_ptrClassifierModel;
  bool m_bUseShapeNet;
};

#pragma once 

#include "GlobalHelpers.h"
#include "CTensorBase.h"
#include "cpu/CTensor.h"
#include "CStringFormatter.h"
#include "cnpy.h"
#include "CPlatformSelection.h"
#include <string>

class CModel1 {
 public:
  CModel1(PLATFORMS targetPlatform,
          unsigned datasetOffset,
          unsigned batchSize,
          unsigned pointsPerPointCloud,
          unsigned knnK,
          bool useShapeNetInstead,
          bool enableOclProfiling,
          bool enableMemBankCrossing,
          bool enableCpuUtilization,
          bool enableTensorDumps);
  ~CModel1();
  void            SetDatasetData(std::string &pathNumpyData);
  void            SetDatasetLabels(std::string &pathNumpyLabels);
  CTensorBasePtr  FullyConnectedForward(CTensorBasePtr inputTn, CTensorBasePtr weightsTn, CTensorBasePtr biasesTn);
  CTensorBasePtr  BatchNormForward(CTensorBasePtr inputTn, CTensorBasePtr gammaTn, CTensorBasePtr betaTn, CTensorBasePtr emaAveTn, CTensorBasePtr emaVarTn);
  CTensorBasePtr  GetEdgeFeatures(CTensorBasePtr inputTn, CTensorBasePtr knnTn);
  CTensorBasePtr  PairwiseDistance(CTensorBasePtr inputTn);
  CTensorBasePtr  TransformNet(CTensorBasePtr edgeFeaturesTn);
  CTensorBasePtr  Execute();
  CTensorBasePtr  GetLabelTn();
  CTensorBasePtr  GetDataTn();
  unsigned        GetBatchSize();
  unsigned        GetClassCount();
  PLATFORMS       GetTargetPlatform();

 private:
  unsigned m_uDatasetOffset=-1;
  unsigned m_uBatchSize=-1;
  unsigned m_uPointsPerCloud=-1;
  unsigned m_uKnnK=-1;
  unsigned m_uClassCount=-1;
  bool m_bUseShapeNet;
  PLATFORMS m_eTargetPlatform;
  CTensorBasePtr m_ptrDatasetDataTn;
  CTensorBasePtr m_ptrDatasetLabelsTn;
  cnpy::NpyArray m_oNumpyObjectData;
  cnpy::NpyArray m_oNumpyObjectLabels;
  CPlatformSelection* m_ptrPlatSelection;
};

 

#pragma once 

#include "GlobalHelpers.h"
#include "CTensorBase.h"
#include "cpu/CTensor.h"
#include "CStringFormatter.h"
#include "cnpy.h"
#include <string>

class CModel1 {
 public:
  CModel1(unsigned datasetOffset, unsigned batchSize, unsigned pointsPerPointCloud, unsigned knnK);
  void            SetDatasetData(std::string &pathNumpyData);
  void            SetDatasetLabels(std::string &pathNumpyLabels);
  CTensorBase*    FullyConnectedForward(CTensorBase* inputTn, CTensorBase* weightsTn, CTensorBase* biasesTn);
  CTensorBase*    BatchNormForward(CTensorBase* inputTn, CTensorBase* gammaTn, CTensorBase* betaTn, CTensorBase* emaAveTn, CTensorBase* emaVarTn);
  CTensorBase*    GetEdgeFeatures(CTensorBase* inputTn, CTensorBase* knnTn);
  CTensorBase*    PairwiseDistance(CTensorBase* inputTn);
  CTensorBase*    TransformNet(CTensorBase* edgeFeaturesTn);
  CTensor<float>* Execute();
  CTensor<int>*   GetLabels();
  unsigned        GetBatchSize();

 private:
  unsigned m_uBatchSize=-1;
  unsigned m_uPointsPerCloud=-1;
  unsigned m_uKnnK=-1;
  CTensorBase* m_ptrDatasetDataTn;
  CTensorBase* m_ptrDatasetLabelsTn;
  unsigned m_uDatasetOffset=0;
  cnpy::NpyArray m_oNumpyObjectData;
  cnpy::NpyArray m_oNumpyObjectLabels;
  //PlatformSelector* platformSelector;
};

 

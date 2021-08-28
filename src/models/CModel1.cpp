#include "models/CModel1.h"

CModel1::CModel1(unsigned datasetOffset, unsigned batchSize, unsigned pointsPerPointCloud, unsigned knnK) {
  m_uClassCount = 40;
  m_uDatasetOffset = datasetOffset;
  m_uBatchSize = batchSize;
  m_uPointsPerCloud = pointsPerPointCloud;
  m_uKnnK = knnK;
}
void CModel1::SetDatasetData(std::string &pathNumpyData) {

}
void CModel1::SetDatasetLabels(std::string &pathNumpyLabels) {

}
CTensorBase *CModel1::FullyConnectedForward(CTensorBase *inputTn, CTensorBase *weightsTn, CTensorBase *biasesTn) {
  return nullptr;
}
CTensorBase *CModel1::BatchNormForward(CTensorBase *inputTn,
                                       CTensorBase *gammaTn,
                                       CTensorBase *betaTn,
                                       CTensorBase *emaAveTn,
                                       CTensorBase *emaVarTn) {
  return nullptr;
}
CTensorBase *CModel1::GetEdgeFeatures(CTensorBase *inputTn, CTensorBase *knnTn) {
  return nullptr;
}
CTensorBase *CModel1::PairwiseDistance(CTensorBase *inputTn) {
  return nullptr;
}
CTensorBase *CModel1::TransformNet(CTensorBase *edgeFeaturesTn) {
  return nullptr;
}
CTensor<float> *CModel1::Execute() {
  return nullptr;
}
CTensor<int> *CModel1::GetLabels() {
  return nullptr;
}
unsigned CModel1::GetBatchSize() {
  return 0;
}
unsigned CModel1::GetClassCount() {
  return m_uClassCount;
}

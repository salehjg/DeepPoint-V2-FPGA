#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "CClassifierMultiPlatform.h"
#include "GlobalHelpers.h"
#include <string>
#include <sys/time.h>

using namespace std;

CClassifierMultiPlatform::CClassifierMultiPlatform(
    bool useShapeNetInstead,
    bool enableOclProfiling,
    bool enableMemBankCrossing,
    bool enableCpuUtilization,
    bool enableTensorDumps){

  m_bUseShapeNet = useShapeNetInstead;
  m_ptrClassifierModel = new CModel1(
      PLATFORMS::XIL,
      0,
      globalBatchsize,
      1024,
      20,
      m_bUseShapeNet,
      enableOclProfiling,
      enableMemBankCrossing,
      enableCpuUtilization,
      enableTensorDumps);
  if(!m_bUseShapeNet){
    string pclPath = globalArgDataPath; pclPath.append("/modelnet40/dataset/dataset_B2048_pcl.npy");
    string labelPath = globalArgDataPath; labelPath.append("/modelnet40/dataset/dataset_B2048_labels_int32.npy");
    SPDLOG_LOGGER_INFO(logger,"PCL NPY PATH: {}", pclPath);
    SPDLOG_LOGGER_INFO(logger,"LBL NPY PATH: {}", labelPath);

    m_ptrClassifierModel->SetDatasetData(pclPath);
    m_ptrClassifierModel->SetDatasetLabels(labelPath);
  }else{
    string pclPath = globalArgDataPath; pclPath.append("/shapenet2/dataset/dataset_B2048_pcl.npy");
    string labelPath = globalArgDataPath; labelPath.append("/shapenet2/dataset/dataset_B2048_labels_int32.npy");
    SPDLOG_LOGGER_INFO(logger,"PCL NPY PATH: {}", pclPath);
    SPDLOG_LOGGER_INFO(logger,"LBL NPY PATH: {}", labelPath);

    m_ptrClassifierModel->SetDatasetData(pclPath);
    m_ptrClassifierModel->SetDatasetLabels(labelPath);
  }

  double timerStart = GetTimestamp();
  auto classScoresTn = m_ptrClassifierModel->Execute();
  SPDLOG_LOGGER_INFO(logger,"Model execution time with batchsize({}): {} Seconds", globalBatchsize, (GetTimestamp() -timerStart));

  CTensorPtr<float> pClassScoresTn = std::dynamic_pointer_cast<CTensor<float>>(classScoresTn);
  CTensorPtr<unsigned> pLabelsTn = std::dynamic_pointer_cast<CTensor<unsigned>>(m_ptrClassifierModel->GetLabelTn());
  CalculateAccuracy(pClassScoresTn, pLabelsTn, m_ptrClassifierModel->GetBatchSize(), 40);
}
double CClassifierMultiPlatform::GetTimestamp() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
void CClassifierMultiPlatform::CalculateAccuracy(CTensorPtr<float> scoresTn,
                                                 CTensorPtr<unsigned> labelsTn,
                                                 unsigned batchSize,
                                                 unsigned classCount) {

  //find argmax(net) and compute bool array of corrects.
  bool *correct = new bool[batchSize];
  float accu =0;

  SPDLOG_LOGGER_INFO(logger,"Computing Accuracy...");
  {
    float max_cte = -numeric_limits<float>::infinity();
    float max = 0;
    unsigned max_indx=-1;
    unsigned *a1 = new unsigned[batchSize];


    for(unsigned b=0;b<batchSize;b++){

      max = max_cte;
      for(unsigned c=0;c<classCount;c++){
        if(max < (*scoresTn)[b*classCount+c]){
          max = (*scoresTn)[b*classCount+c];
          max_indx = c;
        }
      }

      //set maximum score for current batch index
      a1[b]=max_indx;
    }

    for(unsigned b=0;b<batchSize;b++){
      if(a1[b]==(int)(*labelsTn)[b]){
        correct[b]=true;
      }
      else{
        correct[b]=false;
      }
    }

    delete[](a1);
  }
  //----------------------------------------------------------------------------------------
  // compute accuracy using correct array.
  {
    float correct_cnt=0;
    for(unsigned b=0;b<batchSize;b++){
      if(correct[b]) correct_cnt++;
    }
    accu = correct_cnt / (float)batchSize;

    SPDLOG_LOGGER_INFO(logger,"Correct Count: {}", correct_cnt);
    SPDLOG_LOGGER_INFO(logger,"Accuracy: {}", accu);
  }
}

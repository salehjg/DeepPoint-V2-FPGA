#include "CClassifierMultiPlatform.h"
#include "GlobalHelpers.h"
#include <string>
#include <sys/time.h>

using namespace std;

CClassifierMultiPlatform::CClassifierMultiPlatform() {
  m_ptrClassifierModel = new CModel1(0, globalBatchsize,1024,20);
  string pclPath = globalArgDataPath; pclPath.append("/dataset/dataset_B2048_pcl.npy");
  string labelPath = globalArgDataPath; labelPath.append("/dataset/dataset_B2048_labels_int32.npy");

  SPDLOG_LOGGER_INFO(logger,"PCL NPY PATH: {}", pclPath);
  SPDLOG_LOGGER_INFO(logger,"LBL NPY PATH: {}", labelPath);

  m_ptrClassifierModel->SetDatasetData(pclPath);
  m_ptrClassifierModel->SetDatasetLabels(labelPath);

  double timerStart = GetTimestamp();
  CTensor<float>* classScoresTn = m_ptrClassifierModel->Execute();
  SPDLOG_LOGGER_INFO(logger,"Model execution time with batchsize({}): {} Seconds", globalBatchsize, (GetTimestamp() -timerStart));

  CalculateAccuracy(classScoresTn, m_ptrClassifierModel->GetLabels(), m_ptrClassifierModel->GetBatchSize(), 40);
}
double CClassifierMultiPlatform::GetTimestamp() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
void CClassifierMultiPlatform::CalculateAccuracy(CTensor<float> *scoresTn,
                                                 CTensor<int> *labelsTn,
                                                 unsigned batchSize,
                                                 unsigned classCount) {

  //find argmax(net) and compute bool array of corrects.
  bool *correct = new bool[batchSize];
  float accu =0;

  SPDLOG_LOGGER_INFO(logger,"Computing Accuracy...");
  {
    float max_cte = -numeric_limits<float>::infinity();
    float max = 0;
    int max_indx=-1;
    int *a1 = new int[batchSize];


    for(int b=0;b<batchSize;b++){

      max = max_cte;
      for(int c=0;c<classCount;c++){
        if(max   <   scoresTn[b*classCount+c]  ){
          max = scoresTn[b*classCount+c];
          max_indx = c;
        }
      }

      //set maximum score for current batch index
      a1[b]=max_indx;
    }

    for(int b=0;b<batchSize;b++){
      if(a1[b]==(int)labelsTn[b]){
        correct[b]=true;
      }
      else{
        correct[b]=false;
      }
    }

    free(a1);
  }
  //----------------------------------------------------------------------------------------
  // compute accuracy using correct array.
  {
    float correct_cnt=0;
    for(int b=0;b<batchSize;b++){
      if(correct[b]==true) correct_cnt++;
    }
    accu = correct_cnt / (float)batchSize;

    SPDLOG_LOGGER_INFO(logger,"Correct Count: {}", correct_cnt);
    SPDLOG_LOGGER_INFO(logger,"Accuracy: {}", accu);
  }


}

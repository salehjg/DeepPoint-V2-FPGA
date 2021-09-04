#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "cpu/CImplementationCpu.h"
CImplementationCpu::CImplementationCpu(CProfiler *profiler) {
  m_ePlatform = PLATFORMS::CPU;
  m_ptrProfiler = profiler;
  ResetLayerIdCounter(100000);
}
void CImplementationCpu::DumpToNumpyFile(std::string npyFileName, CTensorBasePtr inputTn, std::string npyDumpDir) {
  // The template member functions of a non-template class should be declared and defined in the header file ONLY.
  if(globalDumpTensors){
    ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
    auto inputFloatTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
    auto inputUintTn = std::dynamic_pointer_cast<CTensor<unsigned>>(inputTn);
    if(inputFloatTn!= NULL){
      DumpToNumpyFile<float>(npyFileName, inputFloatTn, npyDumpDir);
    }else if(inputUintTn!= NULL){
      DumpToNumpyFile<unsigned>(npyFileName, inputUintTn, npyDumpDir);
    }else{
      ThrowException("Unsupported tensor type.");
    }
  }
}
bool CImplementationCpu::CompareTensors(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) {
  // The template member functions of a non-template class should be declared and defined in the header file ONLY.
  ValidateTensorPlatforms({inputTn1, inputTn2}, PLATFORMS::CPU);
  auto inputFloatTn1 = std::dynamic_pointer_cast<CTensor<float>>(inputTn1);
  auto inputFloatTn2 = std::dynamic_pointer_cast<CTensor<float>>(inputTn2);
  auto inputUintTn1  = std::dynamic_pointer_cast<CTensor<unsigned>>(inputTn1);
  auto inputUintTn2  = std::dynamic_pointer_cast<CTensor<unsigned>>(inputTn2);
  if(inputFloatTn1!= NULL && inputFloatTn2!= NULL){
    return CompareTensors<float>(inputFloatTn1, inputFloatTn2);
  }else if(inputUintTn1!= NULL && inputUintTn2!= NULL){
    return CompareTensors<unsigned>(inputUintTn1, inputUintTn2);
  }else{
    ThrowException("Unsupported tensor types.");
  }
}
CTensorBasePtr CImplementationCpu::Concat2(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis){
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape1",inputTn1->GetShape()},{"shape2",inputTn2->GetShape()}}),
      new CProfiler::DictIntPtr({{"concatAxis",concatAxis}}),
      nullptr);

  ValidateTensorPlatforms({inputTn1,inputTn2}, PLATFORMS::CPU);
  ConditionCheck(inputTn1->GetRank()==inputTn2->GetRank(), "Input tensors are of unequal ranks.");
  ConditionCheck(inputTn1->GetRank()==4, "Only rank 4 tensors are supported.");

  unsigned rank  = inputTn1->GetRank();
  CTensorPtr<float> rsltTn;

  if(rank==4) {
    auto pInputTn1 = std::dynamic_pointer_cast<CTensor<float>>(inputTn1);
    auto pInputTn2 = std::dynamic_pointer_cast<CTensor<float>>(inputTn2);
    auto shapeTn1 = pInputTn1->GetShape();
    auto shapeTn2 = pInputTn2->GetShape();
    unsigned dimA0 = shapeTn1[0];
    unsigned dimA1 = shapeTn1[1];
    unsigned dimA2 = shapeTn1[2];
    unsigned dimA3 = shapeTn1[3];
    unsigned dimB0 = shapeTn2[0];
    unsigned dimB1 = shapeTn2[1];
    unsigned dimB2 = shapeTn2[2];
    unsigned dimB3 = shapeTn2[3];
    unsigned dimR0 = 0, dimR1 = 0, dimR2 = 0, dimR3 = 0;
    unsigned mat2_offset_dim0 = 0;
    unsigned mat2_offset_dim1 = 0;
    unsigned mat2_offset_dim2 = 0;
    unsigned mat2_offset_dim3 = 0;

    if (concatAxis == 0) {
      dimR0 = dimA0 + dimB0;
      dimR1 = dimA1;
      dimR2 = dimA2;
      dimR3 = dimA3;
      mat2_offset_dim0 = dimA0;
    }
    if (concatAxis == 1) {
      dimR0 = dimA0;
      dimR1 = dimA1 + dimB1;
      dimR2 = dimA2;
      dimR3 = dimA3;
      mat2_offset_dim1 = dimA1;
    }
    if (concatAxis == 2) {
      dimR0 = dimA0;
      dimR1 = dimA1;
      dimR2 = dimA2 + dimB2;
      dimR3 = dimA3;
      mat2_offset_dim2 = dimA2;
    }
    if (concatAxis == 3) {
      dimR0 = dimA0;
      dimR1 = dimA1;
      dimR2 = dimA2;
      dimR3 = dimA3 + dimB3;
      mat2_offset_dim3 = dimA3;
    }

    rsltTn = CTensorPtr<float>(new CTensor<float>({dimR0,dimR1,dimR2,dimR3}));
    unsigned indxS1, indxS2, indxD;

    for (unsigned d0 = 0; d0 < dimA0; d0++) {
      for (unsigned d1 = 0; d1 < dimA1; d1++) {
        for (unsigned d2 = 0; d2 < dimA2; d2++) {
          for (unsigned d3 = 0; d3 < dimA3; d3++) {
            indxS1 = d0 * dimA1 * dimA2 * dimA3 +
                d1 * dimA2 * dimA3 +
                d2 * dimA3 +
                d3;
            indxD = (d0) * dimR1 * dimR2 * dimR3 +
                (d1) * dimR2 * dimR3 +
                (d2) * dimR3 +
                (d3);
            (*rsltTn)[indxD] = (*pInputTn1)[indxS1];
          }
        }
      }
    }

    for (unsigned d0 = 0; d0 < dimB0; d0++) {
      for (unsigned d1 = 0; d1 < dimB1; d1++) {
        for (unsigned d2 = 0; d2 < dimB2; d2++) {
          for (unsigned d3 = 0; d3 < dimB3; d3++) {
            indxS2 = d0 * dimB1 * dimB2 * dimB3 +
                d1 * dimB2 * dimB3 +
                d2 * dimB3 +
                d3;
            indxD = (d0 + mat2_offset_dim0) * dimR1 * dimR2 * dimR3 +
                (d1 + mat2_offset_dim1) * dimR2 * dimR3 +
                (d2 + mat2_offset_dim2) * dimR3 +
                (d3 + mat2_offset_dim3);
            (*rsltTn)[indxD] = (*pInputTn2)[indxS2];
          }
        }
      }
    }
  }
  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::MatMul(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape1",inputTn1->GetShape()},{"shape2",inputTn2->GetShape()}}),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn1,inputTn2}, PLATFORMS::CPU);

  ConditionCheck(inputTn1->GetRank()==inputTn2->GetRank(), "Input tensors are of unequal ranks.");
  ConditionCheck(inputTn1->GetRank()==3||inputTn1->GetRank()==2, "Only rank 3 or rank 2 tensors are supported.");

  auto pInputTn1 = std::dynamic_pointer_cast<CTensor<float>>(inputTn1);
  auto pInputTn2 = std::dynamic_pointer_cast<CTensor<float>>(inputTn2);
  unsigned diff = pInputTn1->ExpandDimZeroToRank(3);
  pInputTn2->ExpandDimZeroToRank(3);

  auto shape1 = pInputTn1->GetShape();
  auto shape2 = pInputTn2->GetShape();
  auto matrixH1  = shape1[1];
  auto matrixW1  = shape1[2];
  auto matrixH2  = shape2[1];
  auto matrixW2  = shape2[2];
  auto batchSize = shape1[0];
  ConditionCheck(matrixW1==matrixH2, "Unequal shape1[2] and shape2[1].");
  ConditionCheck(shape1[0]==shape2[0], "Unequal shape1[0] and shape2[0].");
  CTensorPtr<float> rsltTn(new CTensor<float>({batchSize,matrixH1,matrixW2}));

  size_t indxS1,indxS2,indxD;

  for(unsigned b=0;b<batchSize;b++) {
    // for element of output of matrixH1 x matrixW2
    for(unsigned j=0;j<matrixH1;j++){
      for(unsigned i=0;i<matrixW2;i++){
        //mat1: select row j
        //mat2: select col i
        float sum=0;
        for(unsigned mat1_x=0;mat1_x<matrixW1;mat1_x++)
        {
          indxS1 = b*matrixH1*matrixW1 + j*matrixW1 + mat1_x;
          indxS2 = b*matrixH2*matrixW2 + mat1_x*matrixW2 + i;
          sum += (*pInputTn1)[indxS1] * (*pInputTn2)[indxS2];
        }
        // for element of output of matrixH1 x matrixW2
        indxD = b*matrixH1*matrixW2 + j*matrixW2 + i;
        (*rsltTn)[indxD] = sum;
      }
    }
  }

  pInputTn1->SqueezeDimZeroTimesTry(diff);
  pInputTn2->SqueezeDimZeroTimesTry(diff);
  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::ReLU(CTensorBasePtr inputTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);
  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  ConditionCheck(inputTn->GetLen()!=0, "The input tensor is of length zero!");
  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
  CTensorPtr<float> rsltTn(new CTensor<float>(inputTn->GetShape()));
  const size_t len = inputTn->GetLen();
  for(size_t i=0;i<len;i++){
    (*rsltTn)[i] = ((*pInputTn)[i]>0) ? (*pInputTn)[i] : 0;
  }
  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::Sqrt(CTensorBasePtr inputTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);
  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  ConditionCheck(inputTn->GetLen()!=0, "The input tensor is of length zero!");
  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
  CTensorPtr<float> rsltTn(new CTensor<float>(inputTn->GetShape()));
  const size_t len = inputTn->GetLen();
  for(size_t i=0;i<len;i++){
    (*rsltTn)[i] = sqrt((*pInputTn)[i]);
  }
  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::Square(CTensorBasePtr inputTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);
  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  ConditionCheck(inputTn->GetLen()!=0, "The input tensor is of length zero!");
  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
  CTensorPtr<float> rsltTn(new CTensor<float>(inputTn->GetShape()));
  const size_t len = inputTn->GetLen();
  for(size_t i=0;i<len;i++){
    (*rsltTn)[i] = ((*pInputTn)[i])*((*pInputTn)[i]);
  }
  m_ptrProfiler->FinishLayer();
  return rsltTn;
}

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
CTensorBasePtr CImplementationCpu::BasicOps(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, BASIC_OPS mode) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape1",inputTn1->GetShape()},{"shape2",inputTn2->GetShape()}}),
      new CProfiler::DictIntPtr({{
                                     "mode",
                                     mode==BASIC_OPS::ADD ? 0 : mode==BASIC_OPS::SUB ? 1 : mode==BASIC_OPS::MUL_ELEMENTWISE ? 2 : 3}}),
      nullptr);
  ValidateTensorPlatforms({inputTn1,inputTn2}, PLATFORMS::CPU);
  ConditionCheck(inputTn1->GetRank()>=inputTn2->GetRank(), "The first input tensor's rank cannot be smaller than the second's.");
  ConditionCheck(inputTn1->GetRank()>=1 && inputTn1->GetRank()<=4, "Bad inputTn1 tensor rank.");
  ConditionCheck(inputTn2->GetRank()>=1 && inputTn2->GetRank()<=4, "Bad inputTn2 tensor rank.");
  auto pInputTn1 = std::dynamic_pointer_cast<CTensor<float>>(inputTn1);
  auto pInputTn2 = std::dynamic_pointer_cast<CTensor<float>>(inputTn2);
  unsigned diff = pInputTn1->ExpandDimZeroToRank(4);
  CTensorPtr<float> rsltTn(new CTensor<float>(pInputTn1->GetShape()));
  {
    unsigned indxS1;
    unsigned indxS2;
    unsigned dim0, dim1, dim2, dim3;
    unsigned dim0B, dim1B, dim2B, dim3B;
    int dim0B_IsNotZero, dim1B_IsNotZero, dim2B_IsNotZero, dim3B_IsNotZero;
    auto shape1 = pInputTn1->GetShape();
    auto shape2 = pInputTn2->GetShape();

    dim0 = shape1[0];
    dim1 = shape1[1];
    dim2 = shape1[2];
    dim3 = shape1[3];

    if(pInputTn2->GetRank()==4){
      dim0B=shape2[0];
      dim1B=shape2[1];
      dim2B=shape2[2];
      dim3B=shape2[3];
    }
    if(pInputTn2->GetRank()==3){
      dim0B=0;
      dim1B=shape2[0];
      dim2B=shape2[1];
      dim3B=shape2[2];
    }
    if(pInputTn2->GetRank()==2){
      dim0B=0;
      dim1B=0;
      dim2B=shape2[0];
      dim3B=shape2[1];
    }
    if(pInputTn2->GetRank()==1 && shape2[0]!=1){
      dim0B=0;
      dim1B=0;
      dim2B=0;
      dim3B=shape2[0];
    }else if(pInputTn2->GetRank()==1 && shape2[0]==1){
      dim0B=0;
      dim1B=0;
      dim2B=0;
      dim3B=1; //and rank should be 1 which already is
    }


    int tmp =15>>(4-pInputTn2->GetRank());
    dim0B_IsNotZero = (tmp >> 3) & 1;
    dim1B_IsNotZero = (tmp >> 2) & 1;
    dim2B_IsNotZero = (tmp >> 1) & 1;
    dim3B_IsNotZero = (tmp >> 0) & 1;

    if(pInputTn2->GetRank()==1 && dim0B==0&&dim1B==0&&dim2B==0&&dim3B==1){//scalar value
      dim3B_IsNotZero=0; //force it to be zero, so in the kernel, indxS2 would be zero;
    }

    for(unsigned d0=0;d0<dim0;d0++){
      for(unsigned d1=0;d1<dim1;d1++) {
        for(unsigned d2=0;d2<dim2;d2++) {
          for(unsigned d3=0;d3<dim3;d3++) {
            indxS1 = d0*dim1*dim2*dim3+
                d1*dim2*dim3+
                d2*dim3+
                d3;
            indxS2 = d0 * dim1B * dim2B * dim3B * dim0B_IsNotZero +
                d1 * dim2B * dim3B * dim1B_IsNotZero +
                d2 * dim3B * dim2B_IsNotZero +
                d3 * dim3B_IsNotZero;

            if(mode==BASIC_OPS::ADD)                      //Add
              (*rsltTn)[indxS1] = (*pInputTn1)[indxS1] + (*pInputTn2)[indxS2];
            else if(mode==BASIC_OPS::SUB)                 //Sub
              (*rsltTn)[indxS1] = (*pInputTn1)[indxS1] - (*pInputTn2)[indxS2];
            else if(mode==BASIC_OPS::MUL_ELEMENTWISE)     //Mul (element wise)
              (*rsltTn)[indxS1] = (*pInputTn1)[indxS1] * (*pInputTn2)[indxS2];
            else if(mode==BASIC_OPS::DIV_ELEMENTWISE)     //Div (element wise)
              (*rsltTn)[indxS1] = (*pInputTn1)[indxS1] / (*pInputTn2)[indxS2];
          }
        }
      }
    }

  }
  
  pInputTn1->SqueezeDimZeroTimesTry(diff);
  rsltTn->SqueezeDimZeroTimesTry(diff);

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::BasicOps(CTensorBasePtr inputTn1, float scalar, BASIC_OPS mode) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn1->GetShape()}}),
      nullptr,
      new CProfiler::DictFloatPtr({{"scalar",scalar}}));
  ValidateTensorPlatforms({inputTn1}, PLATFORMS::CPU);

  // This method is used only in CPU impl.
  CTensorBasePtr tmpTn(new CTensor<float>({1}, &scalar));
  auto rsltTn = BasicOps(inputTn1, tmpTn, mode);

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}

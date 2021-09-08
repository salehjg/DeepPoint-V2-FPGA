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
CTensorBasePtr CImplementationCpu::Tile(CTensorBasePtr inputTn, unsigned tileAxis, unsigned tileCount) {
  // inputTn       rsltTn         tileAxis        inputTn's Rank
  // BxNxD   ----> BxNxKxD        2               3
  // BxN     ----> BxNxK          2               2
  // BxN     ----> BxKxN          1               2

  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({{"tileAxis",tileAxis},{"tileCount",tileCount}}),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);

  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
  const unsigned rank = pInputTn->GetRank();
  auto shape = pInputTn->GetShape();
  CTensorPtr<float> rsltTn;
  unsigned indxS1, indxD;

  if(rank==3 && tileAxis==2) {
    unsigned B,N,K,D;
    B = shape[0];
    N = shape[1];
    D = shape[2];
    K = tileCount;

    // Tile input of shape BxNxD into BxNxKxD.
    rsltTn = CTensorPtr<float>(new CTensor<float>({B, N, K, D}));

    for (unsigned b = 0; b < B; b++) {
      for (unsigned n = 0; n < N; n++) {
        indxS1 = b * N * D + n * D + 0; //beginning of dim2 of input
        for (unsigned k = 0; k < K; k++) {
          indxD = b * N * K * D + n * K * D + k * D + 0;
          std::copy(pInputTn->Get() + indxS1,
                    pInputTn->Get() + indxS1 + D,
                    rsltTn->Get() + indxD);
        }
      }
    }

  }

  if(rank==2 && tileAxis==2) { //BxN = BxNx1   ------->  BxNxK  (PAGE 221 of the notebook)
    unsigned B,N,K,D;
    B = shape[0];
    N = shape[1];
    K = tileCount;

    // Tile input of shape BxN or BxNx1 into BxNxK.
    rsltTn = CTensorPtr<float>(new CTensor<float>({B, N, K}));

    for (unsigned b = 0; b < B; b++) {
      for (unsigned n = 0; n < N; n++) {
        indxS1 = b*N + n;
        for(unsigned k=0;k<K;k++){
          indxD = b*N*K + n*K + k;
          (*rsltTn)[indxD] = (*pInputTn)[indxS1];
        }
      }
    }

  }

  if(rank==2 && tileAxis==1) { //BxN = Bx1xN   ------->  BxKxN  (PAGE 221 of my notebook)
    unsigned B,N,K,D;
    B = shape[0];
    N = shape[1];
    K = tileCount;

    // Tile input of shape BxN or Bx1xN into BxKxN.
    rsltTn = CTensorPtr<float>(new CTensor<float>({B, K, N}));

    for(unsigned b = 0; b < B; b++) {
      for(unsigned k=0; k<K; k++){
        for(unsigned n = 0; n < N; n++) {
          indxD  = b*K*N + k*N + n;
          indxS1 = b*1*N + n;
          (*rsltTn)[indxD] = (*pInputTn)[indxS1];
        }
      }
    }

  }

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::Transpose(CTensorBasePtr inputTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  ConditionCheck(inputTn->GetRank()==3 || inputTn->GetRank()==2, "Unsupported input tensor rank.");
  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
  unsigned diff = pInputTn->ExpandDimZeroToRank(3);
  auto shape = pInputTn->GetShape();
  auto dim0 = shape[0];
  auto dim1 = shape[1];
  auto dim2 = shape[2];
  CTensorPtr<float> rsltTn(new CTensor<float>({dim0, dim2, dim1}));
  unsigned indxS, indxD;

  for(unsigned b=0; b<dim0; b++){
    for(unsigned j = 0; j < dim1; j++) {
      for(unsigned i = 0; i < dim2 ; i++) {
        indxS = b * dim1 * dim2 + j * dim2 + i;
        indxD = b * dim1 * dim2 + i * dim1 + j;
        (*rsltTn)[indxD] = (*pInputTn)[indxS];
      }
    }
  }

  pInputTn->SqueezeDimZeroTimesTry(diff);
  ///TODO: Confirm that squeezing the output tensor is NOT required ?
  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::Gather(CTensorBasePtr inputTn, CTensorBasePtr indicesTn, unsigned indicesOfAxis) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  ConditionCheck(inputTn->GetRank()==3, "inputTn is required to be of rank 3.");
  ConditionCheck(indicesTn->GetRank()==3, "indicesTn is required to be of rank 3.");
  ConditionCheck(inputTn->GetShape()[0]==indicesTn->GetShape()[0], "Incompatible shapes.");
  ConditionCheck(inputTn->GetShape()[1]==indicesTn->GetShape()[1], "Incompatible shapes.");
  ConditionCheck(indicesOfAxis==1, "Unsupported indicesOfAxis.");

  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
  auto pIndices = std::dynamic_pointer_cast<CTensor<unsigned>>(indicesTn);


  //inputTn       is considered as BxNxD
  //indices       is considered as BxNxK
  //indices_axis  is considered to be 1 (the dimension that is equal to 'N')

  //Gather knn's indices from input array.
  unsigned indxS1, indxS2, indxD;
  auto shape = pInputTn->GetShape();
  unsigned
      B = shape[0],
      N = shape[1],
      K = pIndices->GetShape()[2],
      D = shape[2];
  CTensorPtr<float> rsltTn(new CTensor<float>({B, N, K, D}));

  for(unsigned b=0;b<B;b++){
    for(unsigned n=0;n<N;n++){
      for(unsigned k=0;k<K;k++){
        indxS1 = b*N*K + n*K + k;
        for(unsigned d=0;d<D;d++){
          indxD = b*N*K*D + n*K*D + k*D + d;
          indxS2 = b*N*D + (*pIndices)[indxS1]*D + d;
          (*rsltTn)[indxD] = (*pInputTn)[indxS2];
        }
      }
    }
  }

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::Reduce(CTensorBasePtr inputTn,
                                          REDUCTION_OPS mode,
                                          unsigned powY,
                                          bool overAxis0,
                                          bool overAxis1,
                                          bool overAxis2,
                                          bool overAxis3) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({
                                    {"reduction_op",
                                     mode==REDUCTION_OPS::SUM?0:
                                     mode==REDUCTION_OPS::MAX?1:
                                     -1
                                    },
                                    {"powY",powY},
                                    {"rank",inputTn->GetRank()},
                                    {"overAxis0",overAxis0},
                                    {"overAxis1",overAxis1},
                                    {"overAxis2",overAxis2},
                                    {"overAxis3",overAxis3}
                                }),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
  ConditionCheck(pInputTn->GetRank()<=4, "The CPU implementation of reduce only supports tensors of rank 4 and less.");

  unsigned diff = 0;
  if(pInputTn->GetRank()<3) {
    diff = pInputTn->ExpandDimZeroToRank(3);
  }
  unsigned rank = pInputTn->GetRank();
  auto shape = pInputTn->GetShape();
  CTensorPtr<float> rsltTn;
  unsigned indxD, indxS;

  if(mode==REDUCTION_OPS::SUM && rank==3){
    //rsltTn = CTensorPtr<float>(new CTensor<float>({B, N, K, D}));

    if(overAxis0&&overAxis1&&overAxis2){ //TTT
      rsltTn = CTensorPtr<float>(new CTensor<float>({1}));
      float sum = 0;
      unsigned limit = pInputTn->GetLen();

      for(unsigned b=0;b<limit;b++) {
        sum += (*pInputTn)[b];
      }
      (*rsltTn)[0] = sum;
    } else if(overAxis0&&!overAxis1&&!overAxis2){ //TFF
      const unsigned dim0 = shape[0], dim1 = shape[1], dim2 = shape[2];
      rsltTn = CTensorPtr<float>(new CTensor<float>({dim1,dim2}));
      float sum=0;

      for(unsigned d1=0; d1<dim1; d1++){
        for(unsigned d2=0; d2<dim2; d2++){
          sum=0;
          indxD = d1 * dim2 + d2;
          for(unsigned dx=0;dx<dim0;dx++){
            indxS = dx * dim1*dim2 + d1 * dim2 + d2;
            sum += (*pInputTn)[indxS] ;
          }
          (*rsltTn)[indxD] = sum;
        }
      }
    }else if(!overAxis0&&overAxis1&&!overAxis2) { //FTF
      const unsigned dim0 = shape[0], dim1 = shape[1], dim2 = shape[2];
      rsltTn = CTensorPtr<float>(new CTensor<float>({dim0,dim2}));
      float sum=0;

      for(unsigned d0=0; d0<dim0;d0++){
        for(unsigned d2=0;d2<dim2;d2++){
          sum=0;
          indxD = d0 *dim2 + d2;
          for(unsigned dx=0;dx<dim1;dx++){
            indxS = d0 * dim1*dim2 + dx * dim2 + d2;
            sum+=(*pInputTn)[indxS] ;
          }
          (*rsltTn)[indxD] = sum;
        }
      }
    }else if(!overAxis0&&!overAxis1&&overAxis2) { //FFT
      const unsigned dim0 = shape[0], dim1 = shape[1], dim2 = shape[2];
      rsltTn = CTensorPtr<float>(new CTensor<float>({dim0,dim1}));
      float sum=0;

      for(unsigned d0=0; d0<dim0;d0++){
        for(unsigned d1=0;d1<dim1;d1++){
          sum=0;
          indxD = d0 * dim1 + d1;
          for(unsigned dx=0;dx<dim2;dx++){
            indxS = d0 * dim1*dim2 + d1 * dim2 + dx;
            sum+=(*pInputTn)[indxS] ;
          }
          (*rsltTn)[indxD] = sum;
        }
      }
    }else{
      ConditionCheck(false, "Unimplemented reduce sum 3 combination.");
    }
  }else if(mode==REDUCTION_OPS::SUM && rank==4) {
    // Sum4 TTTF
    if(overAxis0 && overAxis1 && overAxis2 && !overAxis3) {
      const unsigned dim0 = shape[0], dim1 = shape[1], dim2 = shape[2], dim3 = shape[3];
      rsltTn = CTensorPtr<float>(new CTensor<float>({dim3}));

      float sum = 0;
      for (unsigned d3 = 0; d3 < dim3; d3++) {
        sum = 0;
        indxD = d3;
        for (unsigned d0 = 0; d0 < dim0; d0++) {
          for (unsigned d1 = 0; d1 < dim1; d1++) {
            for (unsigned d2 = 0; d2 < dim2; d2++) {

              indxS = d0 * dim1 * dim2 * dim3 +
                  d1 * dim2 * dim3 +
                  d2 * dim3 +
                  d3;

              sum += (*pInputTn)[indxS];
            }
          }
        }
        (*rsltTn)[indxD] = sum;
      }
    }else{
      ConditionCheck(false, "Unimplemented reduce sum 4 combination.");
    }
  }else if(mode==REDUCTION_OPS::MAX && rank==4){
    // Max4 FTFF
    if(!overAxis0 && !overAxis1 && !overAxis2 && overAxis3){ //over dim 3
      const unsigned dim0 = shape[0], dim1 = shape[1], dim2 = shape[2], dim3 = shape[3];
      rsltTn = CTensorPtr<float>(new CTensor<float>({dim0, dim1, dim2}));

      const float max_cte= -std::numeric_limits<float>::max();
      float max= -std::numeric_limits<float>::max();

      for(unsigned d0=0;d0<dim0;d0++){
        for(unsigned d1=0;d1<dim1;d1++){
          for(unsigned d2=0;d2<dim2;d2++){
            indxD = d0*dim1*dim2+
                d1*dim2+
                d2;
            max = max_cte;
            for(unsigned d3=0;d3<dim3;d3++){
              indxS = d0*dim1*dim2*dim3+
                  d1*dim2*dim3+
                  d2*dim3+
                  d3;
              if(max<(*pInputTn)[indxS]){
                max = (*pInputTn)[indxS];
              }
            }
            (*rsltTn)[indxD]=max;
          }
        }
      }
    }else if(!overAxis0 && !overAxis1 && overAxis2 && !overAxis3) { //over dim 2
      const unsigned dim0 = shape[0], dim1 = shape[1], dim2 = shape[2], dim3 = shape[3];
      rsltTn = CTensorPtr<float>(new CTensor<float>({dim0, dim1, dim3}));

      const float max_cte= -std::numeric_limits<float>::max();
      float max= 0;

      for(unsigned d0=0;d0<dim0;d0++){
        for(unsigned d1=0;d1<dim1;d1++){
          for(unsigned d3=0;d3<dim3;d3++){
            indxD = d0*dim1*dim3+
                d1*dim3+
                d3;
            max = max_cte;

            for(unsigned d2=0;d2<dim2;d2++){
              indxS = d0*dim1*dim2*dim3+
                  d1*dim2*dim3+
                  d2*dim3+
                  d3;
              if(max<(*pInputTn)[indxS]){
                max = (*pInputTn)[indxS];
              }
            }
            (*rsltTn)[indxD]=max;
          }
        }
      }
    }else if(!overAxis0 && overAxis1 && !overAxis2 && !overAxis3) { //over dim 1
      const unsigned dim0 = shape[0], dim1 = shape[1], dim2 = shape[2], dim3 = shape[3];
      rsltTn = CTensorPtr<float>(new CTensor<float>({dim0, dim2, dim3}));

      const float max_cte= -std::numeric_limits<float>::max();
      float max= 0;

      for(unsigned d0=0;d0<dim0;d0++){
        for(unsigned d2=0;d2<dim2;d2++){
          for(unsigned d3=0;d3<dim3;d3++){
            indxD = d0*dim2*dim3+
                d2*dim3+
                d3;
            max = max_cte;

            for(unsigned d1=0;d1<dim1;d1++){
              indxS = d0*dim1*dim2*dim3+
                  d1*dim2*dim3+
                  d2*dim3+
                  d3;
              if(max<(*pInputTn)[indxS]){
                max = (*pInputTn)[indxS];
              }
            }
            (*rsltTn)[indxD]=max;
          }
        }
      }
    }else{
      ConditionCheck(false, "Unimplemented reduce max 4 combination.");
    }

  }else{
    assert(false); //NYI
  }

  pInputTn->SqueezeDimZeroTimesTry(diff);
  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::Mean(CTensorBasePtr inputTn,
                                        bool overAxis0,
                                        bool overAxis1,
                                        bool overAxis2,
                                        bool overAxis3) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({
                                    {"rank",inputTn->GetRank()},
                                    {"overAxis0",overAxis0},
                                    {"overAxis1",overAxis1},
                                    {"overAxis2",overAxis2},
                                    {"overAxis3",overAxis3}
                                }),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);

  auto shape = pInputTn->GetShape();
  unsigned
      dim0 = shape[0],
      dim1 = shape[1],
      dim2 = shape[2],
      dim3 = shape[3],
      rank = pInputTn->GetRank();

  CTensorPtr<float> rsltTn;

  if(rank==4){
    if(!overAxis3 && overAxis0 && overAxis1 && overAxis2){
      CTensorBasePtr reduced = Reduce(inputTn, REDUCTION_OPS::SUM, 1, overAxis0, overAxis1, overAxis2, overAxis3);
      CTensorPtr<float> pReduced = std::dynamic_pointer_cast<CTensor<float>>(reduced);
      rsltTn = CTensorPtr<float>(new CTensor<float>({dim3}));
      const auto l = (float)(dim0*dim1*dim2);
      for(unsigned d3=0;d3<dim3;d3++){
        (*rsltTn)[d3] = ((*pReduced)[d3])/l;
      }
    }
  }

  if(rank==2) { //dim0 is batch, dim1 is fc layer output, ex.: for B=1 --> output=[1,256]
    if (!overAxis1 && overAxis0) {
      CTensorBasePtr reduced = Reduce(inputTn, REDUCTION_OPS::SUM, 1, false, true, false, false);
      auto meanTn = BasicOps(reduced, 1.0f/(float)dim0,BASIC_OPS::MUL_ELEMENTWISE);
      rsltTn = std::dynamic_pointer_cast<CTensor<float>>(meanTn);
    }
  }

  if(rank==1){
    CTensorBasePtr reduced = Reduce(inputTn, REDUCTION_OPS::SUM, 1, true, true, true, false);
    CTensorBasePtr meanTn = BasicOps(reduced, 1.0f/(float)dim0,BASIC_OPS::MUL_ELEMENTWISE);
    rsltTn = std::dynamic_pointer_cast<CTensor<float>>(meanTn);
  }

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::Variance(CTensorBasePtr inputTn,
                                            bool overAxis0,
                                            bool overAxis1,
                                            bool overAxis2,
                                            bool overAxis3) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({
                                    {"rank",inputTn->GetRank()},
                                    {"overAxis0",overAxis0},
                                    {"overAxis1",overAxis1},
                                    {"overAxis2",overAxis2},
                                    {"overAxis3",overAxis3}
                                }),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);

  auto shape = pInputTn->GetShape();
  unsigned
      dim0 = shape[0],
      dim1 = shape[1],
      dim2 = shape[2],
      dim3 = shape[3],
      rank = pInputTn->GetRank();

  CTensorPtr<float> rsltTn;

  if(rank==4){
    if(!overAxis3 && overAxis0 && overAxis1 && overAxis2) {
      CTensorBasePtr meanTn = Mean(inputTn, overAxis0, overAxis1, overAxis2, overAxis3);
      CTensorPtr<float> pMeanTn = std::dynamic_pointer_cast<CTensor<float>>(meanTn);
      CTensorPtr<float> varianceTn(new CTensor<float>({dim3}));
      unsigned indxS1;
      for (unsigned d3 = 0; d3 < dim3; d3++) { //over the last-dim
        (*varianceTn)[d3]=0;


        for (unsigned d0 = 0; d0 < dim0; d0++) {
          for (unsigned d1 = 0; d1 < dim1; d1++) {
            for (unsigned d2 = 0; d2 < dim2; d2++) {
              indxS1 = d0*dim1*dim2*dim3+
                  d1*dim2*dim3+
                  d2*dim3+
                  d3;

              float delta = ((*pInputTn)[indxS1] - (*pMeanTn)[d3]);
              (*varianceTn)[d3] += delta*delta;
            }
          }
        }
      }

      auto varianceFinalTn = BasicOps(varianceTn,(float)(1.0f/(float)(dim0*dim1*dim2)),BASIC_OPS::MUL_ELEMENTWISE);
      rsltTn = std::dynamic_pointer_cast<CTensor<float>>(varianceFinalTn);
    }
  }

  if(rank==2) { //dim0 is batch, dim1 is fc layer output, ex.: for B=1 --> output=[1,256]
    if(!overAxis1 && overAxis0) {
      CTensorBasePtr meanTn = Mean(inputTn, true, false, false, false);
      CTensorPtr<float> pMeanTn = std::dynamic_pointer_cast<CTensor<float>>(meanTn);
      CTensorPtr<float> varianceTn(new CTensor<float>({dim1}));

      unsigned indxS1;
      for(unsigned d1 = 0; d1 < dim1; d1++) { //over the last-dim
        (*varianceTn)[d1]=0;

        for (unsigned d0 = 0; d0 < dim0; d0++) {
          indxS1 = d0*dim1 + d1;

          float delta = ((*pInputTn)[indxS1]-(*pMeanTn)[d1]);
          (*varianceTn)[d1] += delta*delta;
        }
      }
      auto varianceFinalTn = BasicOps(varianceTn,(float)(1.0f/(float)(dim0)),BASIC_OPS::MUL_ELEMENTWISE);
      rsltTn = std::dynamic_pointer_cast<CTensor<float>>(varianceFinalTn);
    }
  }

  if(rank==1){
    CTensorBasePtr meanTn = Mean(inputTn, true, true, true, true);
    CTensorPtr<float> pMeanTn = std::dynamic_pointer_cast<CTensor<float>>(meanTn);
    CTensorPtr<float> varianceTn(new CTensor<float>({1}));

    unsigned indxS1;
    for (int d0 = 0; d0 < dim0; d0++) {
      float delta = ((*pInputTn)[d0] - (*pMeanTn)[0]);
      (*varianceTn)[0] += delta*delta;
    }
    (*varianceTn)[0] = (*varianceTn)[0]/(float)dim0;
    rsltTn = std::dynamic_pointer_cast<CTensor<float>>(varianceTn);
  }

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::PadLastDim(CTensorBasePtr inputTn, unsigned lastDimPadded) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({{"lastDimPadded",lastDimPadded}}),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  ConditionCheck(
      (lastDimPadded>=inputTn->GetShape().back()),
      "lastDimPadded should be greater than shape[-1]."
  );

  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);

  unsigned dim0, dim1, lcm, _gcd;
  auto shape = pInputTn->GetShape();
  const unsigned rank = pInputTn->GetRank();

  if(rank!=1){
    dim0=1;
    for(int i=0; i<rank-1; i++){
      dim0*=shape[i];
    }
    dim1=shape[rank-1];
  }else{
    dim0 = 1;
    dim1 = shape[0];
  }

  if(shape[rank-1]<CONFIG_M_AXI_WIDTH){
    //sub-vector padding
    _gcd = std::__gcd(dim1, CONFIG_M_AXI_WIDTH);
    lcm = (dim1*CONFIG_M_AXI_WIDTH)/(_gcd);
  }else{
    lcm=0;
  }

  shape[rank-1] = lastDimPadded;
  CTensorPtr<float> rsltTn(new CTensor<float>(shape));

  for(unsigned d0=0; d0<dim0; d0++){
    for(unsigned d1=0; d1<lastDimPadded; d1++){
      (*rsltTn)[d0*lastDimPadded+d1] = (d1<dim1) ? (*pInputTn)[d0*dim1+d1] : 0;
    }
  }

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::UnpadLastDim(CTensorBasePtr inputTn, unsigned lastDimUnpadded) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({{"lastDimUnpadded",lastDimUnpadded}}),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  ConditionCheck(
      (lastDimUnpadded<=inputTn->GetShape().back()),
      "lastDimUnpadded should be less than shape[-1]."
  );

  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);

  unsigned dim0, dim1;
  auto shape = pInputTn->GetShape();
  const unsigned rank = pInputTn->GetRank();

  if(rank!=1){
    dim0=1;
    for(int i=0; i<rank-1; i++){
      dim0*=shape[i];
    }
    dim1=shape[rank-1];
  }else{
    dim0 = 1;
    dim1 = shape[0];
  }

  shape[rank-1] = lastDimUnpadded;
  CTensorPtr<float> rsltTn(new CTensor<float>(shape));

  for(unsigned d0=0; d0<dim0; d0++){
    for(unsigned d1=0; d1<lastDimUnpadded; d1++){
      (*rsltTn)[d0*lastDimUnpadded+d1] = (*pInputTn)[d0*dim1+d1];
    }
  }

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}
CTensorBasePtr CImplementationCpu::TopK(CTensorBasePtr inputTn, unsigned axis, unsigned k) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({
        {"axis",axis},
        {"k",k},
        }),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
  ConditionCheck(inputTn->GetRank()==3, "Only input tensors of rank 3 are supported.");
  ConditionCheck(inputTn->GetShape()[2]>k && k>0, "The value for k should be greater than zero and less than shape[2].");
  ConditionCheck(axis==2, "Only axis=2 is supported.");

  auto pInputTn = std::dynamic_pointer_cast<CTensor<float>>(inputTn);
  const auto shape = pInputTn->GetShape();

  unsigned indxS = 0;
  const unsigned B = shape[0], N2 = shape[1], N = shape[2], K = (unsigned)k;
  CTensorPtr<unsigned> rsltTn(new CTensor<unsigned>({B,N2,K}));
  float tmp_array[N];
  unsigned indices[N];

  for(unsigned b=0;b<B*N2;b++){
    for(unsigned i = 0 ;i<N;i++){
      indices[i]=i;
    }
    indxS = b*N + 0;
    std::copy(pInputTn->Get() +indxS, pInputTn->Get()+indxS+N, tmp_array);
    std::sort(  indices,
                indices+N,
                [&](int i1, int i2) { return tmp_array[i1] < tmp_array[i2]; } );

    std::copy(indices, indices+K, rsltTn->Get()+(b*K));
  }

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}

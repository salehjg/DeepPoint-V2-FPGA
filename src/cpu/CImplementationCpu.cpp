#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "cpu/CImplementationCpu.h"
CImplementationCpu::CImplementationCpu(CProfiler *profiler) {
  m_ePlatform = PLATFORMS::CPU;
  m_ptrProfiler = profiler;
  ResetLayerIdCounter(100000);
}
CTensorBase *CImplementationCpu::Concat2(CTensorBase *inputTn1, CTensorBase *inputTn2, int concatAxis) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape1",inputTn1->GetShape()},{"shape2",inputTn2->GetShape()}}),
      new CProfiler::DictIntPtr({{"concatAxis",concatAxis}}),
      nullptr);

  ValidateTensorPlatforms({inputTn1,inputTn2}, PLATFORMS::CPU);
  if(inputTn1->GetRank() != inputTn2->GetRank()){
    throw std::runtime_error(CStringFormatter() << __func__ << ": Input tensors are of unequal ranks.");
  }

  int rank  = inputTn1->GetRank();
  CTensor<float>* rsltTn;

  if(rank==4) {
    CTensor<float>* xinputTn1 = dynamic_cast<CTensor<float>*>(inputTn1);
    CTensor<float>* xinputTn2 = dynamic_cast<CTensor<float>*>(inputTn2);
    auto shapeTn1 = inputTn1->GetShape();
    auto shapeTn2 = inputTn2->GetShape();
    unsigned dimA0 = shapeTn1[0];
    unsigned dimA1 = shapeTn1[1];
    unsigned dimA2 = shapeTn1[2];
    unsigned dimA3 = shapeTn1[3];
    unsigned dimB0 = shapeTn2[0];
    unsigned dimB1 = shapeTn2[1];
    unsigned dimB2 = shapeTn2[2];
    unsigned dimB3 = shapeTn2[3];
    unsigned dimR0 = 0, dimR1 = 0, dimR2 = 0, dimR3 = 0;
    int mat2_offset_dim0 = 0;
    int mat2_offset_dim1 = 0;
    int mat2_offset_dim2 = 0;
    int mat2_offset_dim3 = 0;

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

    rsltTn = new CTensor<float>({dimR0,dimR1,dimR2,dimR3});
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
            (*rsltTn)[indxD] = (*xinputTn1)[indxS1];
          }
        }
      }
    }

    for (int d0 = 0; d0 < dimB0; d0++) {
      for (int d1 = 0; d1 < dimB1; d1++) {
        for (int d2 = 0; d2 < dimB2; d2++) {
          for (int d3 = 0; d3 < dimB3; d3++) {
            indxS2 = d0 * dimB1 * dimB2 * dimB3 +
                d1 * dimB2 * dimB3 +
                d2 * dimB3 +
                d3;
            indxD = (d0 + mat2_offset_dim0) * dimR1 * dimR2 * dimR3 +
                (d1 + mat2_offset_dim1) * dimR2 * dimR3 +
                (d2 + mat2_offset_dim2) * dimR3 +
                (d3 + mat2_offset_dim3);
            (*rsltTn)[indxD] = (*xinputTn2)[indxS2];
          }
        }
      }
    }

  }

  m_ptrProfiler->FinishLayer();
  return rsltTn;
}

void CImplementationCpu::DumpToNumpyFile(std::string npyFileName, CTensorBase *inputTn, std::string npyDumpDir) {
  // The template member functions of a non-template class should be declared and defined in the header file ONLY.
  if(globalDumpTensors){
    ValidateTensorPlatforms({inputTn}, PLATFORMS::CPU);
    auto* inputFloatTn = dynamic_cast<CTensor<float>*>(inputTn);
    auto* inputUintTn = dynamic_cast<CTensor<unsigned>*>(inputTn);
    if(inputFloatTn!= nullptr){
      DumpToNumpyFile<float>(npyFileName, inputFloatTn, npyDumpDir);
    }else if(!inputUintTn){
      DumpToNumpyFile<unsigned>(npyFileName, inputUintTn, npyDumpDir);
    }else{
      throw std::runtime_error(CStringFormatter() << __func__ << ": Unsupported tensor type.");
    }
  }
}

bool CImplementationCpu::CompareTensors(CTensorBase *inputTn1, CTensorBase *inputTn2) {
  // The template member functions of a non-template class should be declared and defined in the header file ONLY.
  ValidateTensorPlatforms({inputTn1, inputTn2}, PLATFORMS::CPU);
  auto* inputFloatTn1 = dynamic_cast<CTensor<float>*>(inputTn1);
  auto* inputFloatTn2 = dynamic_cast<CTensor<float>*>(inputTn2);
  auto* inputUintTn1 = dynamic_cast<CTensor<unsigned>*>(inputTn1);
  auto* inputUintTn2 = dynamic_cast<CTensor<unsigned>*>(inputTn2);
  if(inputFloatTn1!= nullptr && inputFloatTn2!= nullptr){
    return CompareTensors<float>(inputFloatTn1, inputFloatTn2);
  }else if(inputUintTn1!= nullptr && inputUintTn2!= nullptr){
    return CompareTensors<unsigned>(inputUintTn1, inputUintTn2);
  }else{
    throw std::runtime_error(CStringFormatter() << __func__ << ": Unsupported tensor types.");
  }
}

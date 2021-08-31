#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "fpga/xilinx/CTensorXil.h"
#include "test_helpers.h"
#include <vector>

template <int N, int BANK, typename T>
bool TensorXilTestType1(){
  CXilinxInfo *xilInfo = platSelection->GetClassPtrImplementationXilinx()->GetXilInfo();
  CTensor<T> *srcTn = GenerateTensor<T>(0,{N});
  auto *deviceTn = new CTensorXil<T>(xilInfo,*srcTn,BANK);
  auto *dstTn = deviceTn->TransferToHost();

  return CompareCTensors(*srcTn, *dstTn);
}

template <int N, int BANK, typename T>
bool TensorXilTestType2(){
  CTensor<T> *srcTn = GenerateTensor<T>(0,{N});
  auto *deviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::XIL, srcTn);
  auto *dstTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::CPU, deviceTn);
  return CompareCTensors(*srcTn, *dstTn);
}

template <int N, int BANK, typename T>
bool TensorXilTestType3(){
  CTensor<T> *srcTn = GenerateTensor<T>(0,{N});
  auto *deviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::XIL, srcTn);
  auto *dstTn = deviceTn->TransferToHost();

  return CompareCTensors(*srcTn, *dstTn);
}

template <int N, typename T>
bool TensorXilTestCloneBanksType1(int pattern){
  CTensor<T> *srcTn = GenerateTensor<T>(pattern,{N});

  vector<unsigned> vBanks;
#ifdef USEMEMORYBANK0
  vBanks.push_back(0);
#endif
#ifdef USEMEMORYBANK1
  vBanks.push_back(1);
#endif
#ifdef USEMEMORYBANK2
  vBanks.push_back(2);
#endif
#ifdef USEMEMORYBANK3
  vBanks.push_back(3);
#endif
  int err=0;
  CXilinxInfo *xilInfo = platSelection->GetClassPtrImplementationXilinx()->GetXilInfo();

  for(unsigned bankSrc:vBanks){
    auto *deviceSrcTn = new CTensorXil<T>(xilInfo,*srcTn,bankSrc);
    auto *hostSrcTn = deviceSrcTn->TransferToHost();
    for(unsigned bankDest:vBanks) {
      SPDLOG_LOGGER_INFO(logger, "From bank {} to {}", bankSrc, bankDest);
      auto *deviceDstTn = deviceSrcTn->CloneIfNeededToBank(bankDest);
      auto *hostDstTn = deviceDstTn->TransferToHost();
      if (err > 0) {
        SPDLOG_LOGGER_ERROR(logger, "Tensor data mismatch");
      }
      err += (int) !CompareCTensors(*hostSrcTn,*hostDstTn);
    }
  }

  return err==0;
}

TEST(test_ctensorxil, type1) {
  std::vector<bool> results = {
#ifdef USEMEMORYBANK0
      TensorXilTestType1<1024,0,float>(),
      TensorXilTestType1<1024,0,unsigned>(),
#endif
#ifdef USEMEMORYBANK1
      TensorXilTestType1<1024,1,float>(),
      TensorXilTestType1<1024,1,unsigned>(),
#endif
#ifdef USEMEMORYBANK2
      TensorXilTestType1<1024,2,float>(),
      TensorXilTestType1<1024,2,unsigned>(),
#endif
#ifdef USEMEMORYBANK3
      TensorXilTestType1<1024,3,float>(),
      TensorXilTestType1<1024,3,unsigned>()
#endif
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ctensorxil, type2) {
  std::vector<bool> results = {
#ifdef USEMEMORYBANK0
      TensorXilTestType2<1024,0,float>(),
      TensorXilTestType2<1024,0,unsigned>(),
#endif
#ifdef USEMEMORYBANK1
      TensorXilTestType2<1024,1,float>(),
      TensorXilTestType2<1024,1,unsigned>(),
#endif
#ifdef USEMEMORYBANK2
      TensorXilTestType2<1024,2,float>(),
      TensorXilTestType2<1024,2,unsigned>(),
#endif
#ifdef USEMEMORYBANK3
      TensorXilTestType2<1024,3,float>(),
      TensorXilTestType2<1024,3,unsigned>()
#endif
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ctensorxil, type3) {
  std::vector<bool> results = {
#ifdef USEMEMORYBANK0
      TensorXilTestType3<1024,0,float>(),
      TensorXilTestType3<1024,0,unsigned>(),
#endif
#ifdef USEMEMORYBANK1
      TensorXilTestType3<1024,1,float>(),
      TensorXilTestType3<1024,1,unsigned>(),
#endif
#ifdef USEMEMORYBANK2
      TensorXilTestType3<1024,2,float>(),
      TensorXilTestType3<1024,2,unsigned>(),
#endif
#ifdef USEMEMORYBANK3
      TensorXilTestType3<1024,3,float>(),
      TensorXilTestType3<1024,3,unsigned>()
#endif
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ctensorxil, padunpad1) {
  std::vector<bool> results = {
#ifdef USEMEMORYBANK0
      TensorXilTestType3<63,0,float>(),
      TensorXilTestType3<63,0,unsigned>(),
#endif
#ifdef USEMEMORYBANK1
      TensorXilTestType3<49,1,float>(),
      TensorXilTestType3<49,1,unsigned>(),
#endif
#ifdef USEMEMORYBANK2
      TensorXilTestType3<62,2,float>(),
      TensorXilTestType3<62,2,unsigned>(),
#endif
#ifdef USEMEMORYBANK3
      TensorXilTestType3<10,3,float>(),
      TensorXilTestType3<10,3,unsigned>()
#endif
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ctensorxil, clonebanks1) {
  std::vector<bool> results = {
    TensorXilTestCloneBanksType1<63,float>(0),
    TensorXilTestCloneBanksType1<63,float>(1),
    TensorXilTestCloneBanksType1<63,float>(2),
    TensorXilTestCloneBanksType1<63,float>(3),
    TensorXilTestCloneBanksType1<63,float>(7),
    TensorXilTestCloneBanksType1<63,unsigned>(0),
    TensorXilTestCloneBanksType1<63,unsigned>(1),
    TensorXilTestCloneBanksType1<63,unsigned>(2),
    TensorXilTestCloneBanksType1<63,unsigned>(3),
    TensorXilTestCloneBanksType1<63,unsigned>(7)
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
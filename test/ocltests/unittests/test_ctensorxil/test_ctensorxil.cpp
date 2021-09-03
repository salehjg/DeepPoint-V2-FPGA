#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "fpga/xilinx/CTensorXil.h"
#include "test_helpers.h"
#include <vector>

template <int N, int BANK, typename T>
bool TensorXilTestType1(){
  CXilinxInfo *xilInfo = platSelection->GetClassPtrImplementationXilinx()->GetXilInfo();
  auto srcTn = GenerateTensor<T>(0,{N});
  auto deviceTn = CTensorXilPtr<T>(new CTensorXil<T>(xilInfo,*srcTn,BANK));
  auto dstTn = deviceTn->TransferToHost();

  return platSelection->CompareTensors(PLATFORMS::CPU,
                                       Convert2TnBasePtr(srcTn),
                                       Convert2TnBasePtr(dstTn)
      );
}

template <int N, int BANK, typename T>
bool TensorXilTestType2(){
  auto srcTn = GenerateTensor<T>(0,{N});
  auto deviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::XIL, Convert2TnBasePtr(srcTn));
  auto dstTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::CPU, deviceTn);
  return platSelection->CompareTensors(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), dstTn);
}

template <int N, int BANK, typename T>
bool TensorXilTestType3(){
  auto srcTn = GenerateTensor<T>(0,{N});
  auto deviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::XIL, Convert2TnBasePtr(srcTn));
  auto dstTn = Convert2TnXilPtr<T>(deviceTn)->TransferToHost();

  return platSelection->CompareTensors(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), Convert2TnBasePtr(dstTn) );
}

template <int N, int BANK, typename T>
bool TensorXilTestType4(){
  auto srcTn = GenerateTensor<T>(0,{N});
  auto deviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::XIL, Convert2TnBasePtr(srcTn));
  auto dstTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::CPU, deviceTn);

  return platSelection->CompareTensors(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), dstTn);
}

template <int N, typename T>
bool TensorXilTestType5(){
  // Redundant platform crossing tests
  // CPU->CPU
  // XIL->XIL (on the default bank)
  auto srcHostTn = GenerateTensor<T>(0,{N});
  auto srcDeviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS::XIL, Convert2TnBasePtr(srcHostTn));

  auto dstTn1 = platSelection->CrossThePlatformIfNeeded(PLATFORMS::CPU, Convert2TnBasePtr(srcHostTn));
  bool cmp1 = platSelection->CompareTensors(PLATFORMS::CPU, Convert2TnBasePtr(srcHostTn), dstTn1);

  auto dstTn2 = platSelection->CrossThePlatformIfNeeded(PLATFORMS::XIL, srcDeviceTn);
  bool cmp2 = platSelection->CompareTensors(PLATFORMS::CPU, srcDeviceTn, dstTn2);

  return cmp1 && cmp2;
}

template <int N, typename T>
bool TensorXilTestType6(){
  // Mixed platform tensor comparisons
  auto srcHostTn = GenerateTensor<T>(0,{N});
  auto srcDeviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS::XIL, Convert2TnBasePtr(srcHostTn));
  bool cmp1 = platSelection->CompareTensors(PLATFORMS::CPU, Convert2TnBasePtr(srcHostTn), srcDeviceTn);
  return cmp1;
}

template <int N, typename T>
bool TensorXilTestCloneBanksType1(int pattern){
  auto srcTn = GenerateTensor<T>(0,{N});

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
    CTensorXilPtr<T> deviceSrcTn(new CTensorXil<T>(xilInfo,*srcTn,bankSrc));
    auto hostSrcTn = deviceSrcTn->TransferToHost();
    for(unsigned bankDest:vBanks) {
      SPDLOG_LOGGER_INFO(logger, "From bank {} to {}", bankSrc, bankDest);
      auto deviceDstTn = deviceSrcTn->CloneIfNeededToBank(bankDest);
      auto hostDstTn = deviceDstTn->TransferToHost();
      if (err > 0) {
        SPDLOG_LOGGER_ERROR(logger, "Tensor data mismatch");
      }
      err += (int) !platSelection->CompareTensors(PLATFORMS::CPU, Convert2TnBasePtr(hostSrcTn), Convert2TnBasePtr(hostDstTn));
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

TEST(test_ctensorxil, type4) {
  std::vector<bool> results = {
#ifdef USEMEMORYBANK0
      TensorXilTestType4<1024,0,float>(),
      TensorXilTestType4<1024,0,unsigned>(),
#endif
#ifdef USEMEMORYBANK1
      TensorXilTestType4<1024,1,float>(),
      TensorXilTestType4<1024,1,unsigned>(),
#endif
#ifdef USEMEMORYBANK2
      TensorXilTestType4<1024,2,float>(),
      TensorXilTestType4<1024,2,unsigned>(),
#endif
#ifdef USEMEMORYBANK3
      TensorXilTestType4<1024,3,float>(),
      TensorXilTestType4<1024,3,unsigned>()
#endif
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ctensorxil, reduntantplatformcrossing1) {

  std::vector<bool> results = {
      TensorXilTestType5<31, float>(),
      TensorXilTestType5<31, unsigned>()
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ctensorxil, mixedplatformtensorcomparison) {

  std::vector<bool> results = {
      TensorXilTestType6<31, float>(),
      TensorXilTestType6<31, unsigned>()
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

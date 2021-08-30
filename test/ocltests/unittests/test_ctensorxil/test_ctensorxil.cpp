#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "fpga/xilinx/CTensorXil.h"
#include "test_helpers.h"
#include <vector>

template <int N, int BANK, typename T>
bool TensorXilTestType1(){
  CXilinxInfo *xilInfo = platSelection->GetClassPtrImplementationXilinx()->GetXilInfo();
  CTensor<T> srcTn({N});
  for(int i=0;i<srcTn.GetLen();i++){srcTn[i]=i;}
  auto *deviceTn = new CTensorXil<T>(xilInfo,srcTn,BANK);
  auto *dstTn = deviceTn->TransferToHost();

  return CompareCTensors(srcTn, *dstTn);
}

template <int N, int BANK, typename T>
bool TensorXilTestType2(){
  CTensor<T> srcTn({N});
  for(int i=0;i<srcTn.GetLen();i++){srcTn[i]=i;}
  auto *deviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::XIL, &srcTn);
  auto *dstTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::CPU, deviceTn);
  return CompareCTensors(srcTn, *dstTn);
}

template <int N, int BANK, typename T>
bool TensorXilTestType3(){
  CTensor<T> srcTn({N});
  for(int i=0;i<srcTn.GetLen();i++){srcTn[i]=i;}
  auto *deviceTn = platSelection->CrossThePlatformIfNeeded(PLATFORMS ::XIL, &srcTn);
  auto *dstTn = deviceTn->TransferToHost();

  return CompareCTensors(srcTn, *dstTn);
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
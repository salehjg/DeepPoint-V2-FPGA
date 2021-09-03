#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool ConcatTestType1(const std::vector<unsigned> &shape1, const std::vector<unsigned> &shape2){
  assert(shape1.size()==4);
  assert(shape2.size()==4);
  auto srcTn1 = GenerateTensor<T>(0,shape1);
  auto srcTn2 = GenerateTensor<T>(0,shape2);
  auto goldTn = platSelection->Concat2(PLATFORMS::CPU, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2), 3);
  auto dstTn = platSelection->Concat2(PLATFORMS::XIL, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2), 3);

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_ckwconcat, type1) {
  std::vector<bool> results = {
      ConcatTestType1<float>({1,1,2,2},{1,1,2,6}),    //sub-vec
      ConcatTestType1<float>({2,1,1,64},{2,1,1,16})   //super-vec
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ckwconcat, dependencytest1) {
  auto srcTn1 = GenerateTensor<float>(0,{2,1,1,16});
  auto srcTn2 = GenerateTensor<float>(0,{2,1,1,32});
  auto tempTn = platSelection->Concat2(PLATFORMS::CPU, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2), 3);
  auto goldTn = platSelection->Concat2(PLATFORMS::CPU, tempTn, Convert2TnBasePtr(srcTn2), 3);

  auto tempTn2 = platSelection->Concat2(PLATFORMS::XIL, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2), 3);
  // The layer below should be run when tempTn2 is ready.
  auto dstTn = platSelection->Concat2(PLATFORMS::XIL, tempTn2, Convert2TnBasePtr(srcTn2), 3);

  bool cmp = platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
  EXPECT_TRUE(cmp);
}

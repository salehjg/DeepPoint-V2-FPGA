#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool ConcatTestType1(const std::vector<unsigned> &shape){
  assert(shape.size()==4);
  CTensor<T> *srcTn1 = GenerateTensor<T>(0,shape);
  CTensor<T> *srcTn2 = GenerateTensor<T>(0,shape);
  auto *goldTn = platSelection->Concat2(PLATFORMS::CPU, srcTn1, srcTn2, 3);
  auto *dstTn = platSelection->Concat2(PLATFORMS::XIL, srcTn1, srcTn2, 3);

  return platSelection->CompareTensors(PLATFORMS::CPU, (CTensorBase*)(goldTn), (CTensorBase*)(dstTn));
}

TEST(test_ckwconcat, type1) {
  std::vector<bool> results = {
      ConcatTestType1<float>({2,2,3,15}),
      ConcatTestType1<float>({2,2,3,16}),
      ConcatTestType1<float>({1,2,3,8}),
      ConcatTestType1<unsigned>({2,2,3,15}),
      ConcatTestType1<unsigned>({2,2,3,16}),
      ConcatTestType1<unsigned>({1,2,3,8}),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

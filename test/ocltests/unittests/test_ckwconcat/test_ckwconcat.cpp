#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool ConcatTestType1(const std::vector<unsigned> &shape1, const std::vector<unsigned> &shape2){
  assert(shape1.size()==4);
  assert(shape2.size()==4);
  CTensor<T> *srcTn1 = GenerateTensor<T>(0,shape1);
  CTensor<T> *srcTn2 = GenerateTensor<T>(0,shape2);
  auto *goldTn = platSelection->Concat2(PLATFORMS::CPU, srcTn1, srcTn2, 3);
  auto *dstTn = platSelection->Concat2(PLATFORMS::XIL, srcTn1, srcTn2, 3);

  return platSelection->CompareTensors(PLATFORMS::CPU, (CTensorBase*)(goldTn), (CTensorBase*)(dstTn));
}

TEST(test_ckwconcat, type1) {
  std::vector<bool> results = {
      ConcatTestType1<float>({1,1,2,2},{1,1,2,6}), //sub-vec
      ConcatTestType1<float>({2,1,1,64},{2,1,1,16}) //super-vec
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

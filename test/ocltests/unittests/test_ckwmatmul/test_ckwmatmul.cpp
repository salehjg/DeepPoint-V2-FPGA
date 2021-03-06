#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool MatmulTestRank3(unsigned sizeBatch, unsigned sizeN, unsigned sizeM, unsigned sizeK){
  auto srcTn1 = GenerateTensor<T>(0,{sizeBatch, sizeN, sizeK});
  auto srcTn2 = GenerateTensor<T>(0,{sizeBatch, sizeK, sizeM});
  auto goldTn = platSelection->MatMul(PLATFORMS::CPU, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2));
  auto dstTn = platSelection->MatMul(PLATFORMS::XIL, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2));

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

template <typename T>
bool MatmulTestRank2(unsigned sizeN, unsigned sizeM, unsigned sizeK){
  auto srcTn1 = GenerateTensor<T>(0,{sizeN, sizeK});
  auto srcTn2 = GenerateTensor<T>(0,{sizeK, sizeM});
  auto goldTn = platSelection->MatMul(PLATFORMS::CPU, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2));
  auto dstTn = platSelection->MatMul(PLATFORMS::XIL, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2));

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_ckwmatmul, rank3) {
  std::vector<bool> results = {
      //MatmulTestType1<float>(1,64,64,64),
      MatmulTestRank3<float>(1,3,5,64),
      MatmulTestRank3<float>(1,4,16,16),
      MatmulTestRank3<float>(1,3,5,4),
      MatmulTestRank3<float>(5,16,3,3),
      //MatmulTestType1<float>(1,17,15,31)
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ckwmatmul, rank2) {
  std::vector<bool> results = {
      MatmulTestRank2<float>(3,2,32),
      MatmulTestRank2<float>(4,16,16),
      MatmulTestRank2<float>(3,5,4)
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
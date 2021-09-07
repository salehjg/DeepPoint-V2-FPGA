#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool VarianceTest(const std::vector<unsigned> &shape){
  auto srcTn = GenerateTensor<T>(0,shape);
  auto goldTn = platSelection->Variance(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), true, true, true, false);
  auto dstTn = platSelection->Variance(PLATFORMS::XIL, Convert2TnBasePtr(srcTn), true, true, true, false);

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_layervariance, variance4_TTTF1) {
  std::vector<bool> results = {
      VarianceTest<float>({2,2,2,5}),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
/*
TEST(test_layervariance, variance4_TTTF2) {
  std::vector<bool> results = {
      VarianceTest<float>({2,2,2,17}),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
*/
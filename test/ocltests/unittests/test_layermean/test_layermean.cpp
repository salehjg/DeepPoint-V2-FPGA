#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool MeanTest(const std::vector<unsigned> &shape){
  auto srcTn = GenerateTensor<T>(0,shape);
  auto goldTn = platSelection->Mean(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), true, true, true, false);
  auto dstTn = platSelection->Mean(PLATFORMS::XIL, Convert2TnBasePtr(srcTn), true, true, true, false);

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_layermean, mean4_TTTF1) {
  std::vector<bool> results = {
      MeanTest<float>({2,2,2,5}),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_layermean, mean4_TTTF2) {
  std::vector<bool> results = {
      MeanTest<float>({2,2,2,17}),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool ConvTest1(
    const std::vector<unsigned> &shapeInput,
    const std::vector<unsigned> &shapeWeight,
    const std::vector<unsigned> &shapeBias){

  auto inputTn = GenerateTensor<T>(0,shapeInput);
  auto weightTn = GenerateTensor<T>(0,shapeWeight);
  auto biasTn = GenerateTensor<T>(0,shapeBias);

  auto goldTn = platSelection->Conv2D(PLATFORMS::CPU,
                                     Convert2TnBasePtr(inputTn),
                                     Convert2TnBasePtr(weightTn),
                                     Convert2TnBasePtr(biasTn));
  auto dstTn = platSelection->Conv2D(PLATFORMS::XIL,
                                     Convert2TnBasePtr(inputTn),
                                     Convert2TnBasePtr(weightTn),
                                     Convert2TnBasePtr(biasTn));

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_ckwconv, mixed1) {
  std::vector<bool> results = {
      ConvTest1<float>({1,256,1,6},{1,1,6,16},{16}),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
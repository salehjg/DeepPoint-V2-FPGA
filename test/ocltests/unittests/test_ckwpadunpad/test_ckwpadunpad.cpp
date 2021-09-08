#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool PadUnpadTest(const std::vector<unsigned> &shape, const unsigned lastDimPadded){
  ConditionCheck(shape.back()>=CONFIG_M_AXI_WIDTH,
      "Sub-vec padding/unpadding is disbaled in the kernel. The CTensorXil is responsible for that now.");
  auto srcTn = GenerateTensor<T>(0,shape);
  auto paddedTn = platSelection->PadLastDim(PLATFORMS::XIL, Convert2TnBasePtr(srcTn), lastDimPadded);
  auto dstTn = platSelection->UnpadLastDim(PLATFORMS::XIL, paddedTn, shape.back());

  return platSelection->CompareTensors(PLATFORMS::CPU, srcTn, dstTn);
}

TEST(test_ckwpadunpad, mixed1) {
  std::vector<bool> results = {
      PadUnpadTest<float>({2,1,2,16}, 32),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

TEST(test_ckwpadunpad, mixed2) {
  std::vector<bool> results = {
      PadUnpadTest<float>({2,1,2,16}, 64),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

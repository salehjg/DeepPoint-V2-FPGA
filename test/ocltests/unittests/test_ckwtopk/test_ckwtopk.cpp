#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool TopkTest(const std::vector<unsigned> &shape, unsigned axis, unsigned k){
  auto srcTn = GenerateTensor<T>(0,shape);
  auto goldTn = platSelection->TopK(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), axis, k);
  auto dstTn = platSelection->TopK(PLATFORMS::XIL, Convert2TnBasePtr(srcTn), axis, k);

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_ckwtopk, mixed1) {
  std::vector<bool> results = {
      TopkTest<float>({1,2,1024}, 2, 20),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

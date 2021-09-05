#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool TransposeTest(const std::vector<unsigned> &shape){
  auto srcTn = GenerateTensor<T>(0,shape);
  auto goldTn = platSelection->Transpose(PLATFORMS::CPU, Convert2TnBasePtr(srcTn));
  auto dstTn = platSelection->Transpose(PLATFORMS::XIL, Convert2TnBasePtr(srcTn));

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_ckwtile, mixed1) {
  std::vector<bool> results = {
      TransposeTest<float>({2,32,32}),
      TransposeTest<float>({64,64}),
      TransposeTest<float>({1,64,64})
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

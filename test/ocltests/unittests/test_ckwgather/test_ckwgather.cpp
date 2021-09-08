#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

bool GatherTest(const std::vector<unsigned> &shapeInputTn, const std::vector<unsigned> &shapeIdicesTn, unsigned indicesOfAxis){
  auto srcTn = GenerateTensor<float>(7,shapeInputTn);
  auto indicesTn = GenerateTensor<unsigned>(0, (float)shapeInputTn[1],shapeIdicesTn);
  auto goldTn = platSelection->Gather(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), Convert2TnBasePtr(indicesTn), indicesOfAxis);
  auto dstTn = platSelection->Gather(PLATFORMS::XIL, Convert2TnBasePtr(srcTn), Convert2TnBasePtr(indicesTn), indicesOfAxis);
  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_ckwgather, mixed1) {
  std::vector<bool> results = {
      GatherTest({5,5,2}, {5,5,3}, 1)
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

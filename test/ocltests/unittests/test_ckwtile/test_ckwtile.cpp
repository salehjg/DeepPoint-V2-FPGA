#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool TileTest(const std::vector<unsigned> &shape, unsigned tileAxis, unsigned tileCount){
  auto srcTn = GenerateTensor<T>(0,shape);
  auto goldTn = platSelection->Tile(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), tileAxis, tileCount);
  auto dstTn = platSelection->Tile(PLATFORMS::XIL, Convert2TnBasePtr(srcTn), tileAxis, tileCount);

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_ckwtile, mixed1) {
  std::vector<bool> results = {
      TileTest<float>({2,2,17}, 2, 8),
      TileTest<float>({2,5}, 2, 7),
      TileTest<float>({2,7}, 1, 3),
      TileTest<float>({2,18}, 1, 2)
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

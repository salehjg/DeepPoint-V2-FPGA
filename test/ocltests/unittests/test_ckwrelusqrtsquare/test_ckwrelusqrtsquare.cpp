#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool RssTestRelu(const std::vector<unsigned> &shape){
  auto srcTn = GenerateTensor<T>(0,shape);
  auto goldTn = platSelection->ReLU(PLATFORMS::CPU, Convert2TnBasePtr(srcTn));
  auto dstTn = platSelection->ReLU(PLATFORMS::XIL, Convert2TnBasePtr(srcTn));

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}
template <typename T>
bool RssTestSqrt(const std::vector<unsigned> &shape){
  auto srcTn = GenerateTensor<T>(0,shape);
  auto goldTn = platSelection->Sqrt(PLATFORMS::CPU, Convert2TnBasePtr(srcTn));
  auto dstTn = platSelection->Sqrt(PLATFORMS::XIL, Convert2TnBasePtr(srcTn));

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}
template <typename T>
bool RssTestSquare(const std::vector<unsigned> &shape){
  auto srcTn = GenerateTensor<T>(0,shape);
  auto goldTn = platSelection->Square(PLATFORMS::CPU, Convert2TnBasePtr(srcTn));
  auto dstTn = platSelection->Square(PLATFORMS::XIL, Convert2TnBasePtr(srcTn));

  return platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
}

TEST(test_ckwrelusqrtsquare, relu1) {
  std::vector<bool> results = {
      RssTestRelu<float>({15}),
      RssTestRelu<float>({33})
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwrelusqrtsquare, sqrt1) {
  std::vector<bool> results = {
      RssTestSqrt<float>({15}),
      RssTestSqrt<float>({33})
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwrelusqrtsquare, square1) {
  std::vector<bool> results = {
      RssTestSquare<float>({15}),
      RssTestSquare<float>({33})
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
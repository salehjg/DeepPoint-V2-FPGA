#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

bool MultiPlatformTestType1(const std::vector<unsigned> &shape1){
  CTensorBasePtr rsltTn1;
  CTensorBasePtr rsltTn2;

  auto _biasCpu = GenerateTensor<float>(0,shape1);
  auto _weightCpu = GenerateTensor<float>(0,shape1);
  auto _initCpu = GenerateTensor<float>(0,shape1);
  {
    PLATFORMS targetPlatform = PLATFORMS::CPU;
    auto _bias = platSelection->CrossThePlatformIfNeeded(targetPlatform, _biasCpu);
    auto _weight = platSelection->CrossThePlatformIfNeeded(targetPlatform, _weightCpu);
    auto tmpTn1 = platSelection->BasicOps(targetPlatform, _bias, Convert2TnBasePtr(_initCpu), BASIC_OPS::ADD);
    platSelection->DumpToNumpyFile(PLATFORMS::CPU, "test_gold1.npy", tmpTn1);
    auto tmpTn2 = platSelection->MatMul(targetPlatform, _bias, _weight);
    platSelection->DumpToNumpyFile(PLATFORMS::CPU, "test_gold2.npy", tmpTn2);
    rsltTn1 = platSelection->BasicOps(targetPlatform, tmpTn1, tmpTn2, BASIC_OPS::ADD);
    platSelection->DumpToNumpyFile(PLATFORMS::CPU, "test_gold3.npy", rsltTn1);
  }

  {
    PLATFORMS targetPlatform = PLATFORMS::XIL;
    auto _biasXil = platSelection->CrossThePlatformIfNeeded(targetPlatform, _biasCpu);
    auto _weightXil = platSelection->CrossThePlatformIfNeeded(targetPlatform, _weightCpu);
    auto tmpTn1 = platSelection->BasicOps(targetPlatform, _biasXil, Convert2TnBasePtr(_initCpu), BASIC_OPS::ADD);
    platSelection->DumpToNumpyFile(PLATFORMS::CPU, "test_uut1.npy", tmpTn1);
    auto tmpTn2 = platSelection->MatMul(targetPlatform, _biasXil, _weightXil);
    platSelection->DumpToNumpyFile(PLATFORMS::CPU, "test_uut2.npy", tmpTn2);
    rsltTn2 = platSelection->BasicOps(targetPlatform, tmpTn1, tmpTn2, BASIC_OPS::ADD);
    platSelection->DumpToNumpyFile(PLATFORMS::CPU, "test_uut3.npy", rsltTn2);
  }

  return platSelection->CompareTensors(PLATFORMS::CPU, rsltTn1, rsltTn2);
}

TEST(test_multiplatform1, senario1) {
  auto r = MultiPlatformTestType1({3,3});
  EXPECT_TRUE(r);
}

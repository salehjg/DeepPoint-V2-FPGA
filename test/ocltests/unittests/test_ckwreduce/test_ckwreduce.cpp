#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool ReduceTest(const unsigned pattern, const std::vector<unsigned> &shape, REDUCTION_OPS op, unsigned powY, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3){
  auto srcTn = GenerateTensor<T>(pattern,shape);
  auto goldTn = platSelection->Reduce(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), op, powY, overAxis0, overAxis1, overAxis2, overAxis3);
  auto dstTn = platSelection->Reduce(PLATFORMS::XIL, Convert2TnBasePtr(srcTn), op, powY, overAxis0, overAxis1, overAxis2, overAxis3);

  bool cmp = platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
  if(!cmp)SPDLOG_LOGGER_TRACE(logger, "Reduce, Rank{}: {}", srcTn->GetRank(), cmp?"PASS":"FAIL");
}

TEST(test_ckwreduce, mixed1) {
  std::vector<bool> results = {
      ReduceTest<float>(0, {1,1024,3}, REDUCTION_OPS::SUM, 1, false, false, true, false),   // RS3 FFT
      ReduceTest<float>(0, {1,1024,64}, REDUCTION_OPS::SUM, 1, false, false, true, false),  // RS3 FFT

      ReduceTest<float>(3, {2,2,16,512}, REDUCTION_OPS::SUM, 1, true, true, true, false),   // RS4 TTTF
      ReduceTest<float>(4, {2,2,2,16}, REDUCTION_OPS::SUM, 1, true, true, true, false),     // RS4 TTTF

      ReduceTest<float>(0, {2,2,3,32}, REDUCTION_OPS::MAX, 1, false, false, true, false),   // RM4 FFTF
      ReduceTest<float>(0, {2,2,3,16}, REDUCTION_OPS::MAX, 1, false, false, true, false),   // RM4 FFTF
      ReduceTest<float>(0, {2,1,1,1024}, REDUCTION_OPS::MAX, 1, false, false, true, false), // RM4 FFTF
      ReduceTest<float>(0, {2,2,1,1024}, REDUCTION_OPS::MAX, 1, false, true, false, false), // RM4 FTFF
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}

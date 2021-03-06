#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool ReduceTest(const unsigned pattern, const std::vector<unsigned> &shape, REDUCTION_OPS op, unsigned powY, const std::vector<unsigned> &combination){
  auto srcTn = GenerateTensor<T>(pattern,shape);
  auto goldTn = platSelection->Reduce(PLATFORMS::CPU, Convert2TnBasePtr(srcTn), op, powY, combination);
  auto dstTn = platSelection->Reduce(PLATFORMS::XIL, Convert2TnBasePtr(srcTn), op, powY, combination);

  bool cmp = platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
  if(!cmp)SPDLOG_LOGGER_TRACE(logger, "Reduce, Rank{}: {}", srcTn->GetRank(), cmp?"PASS":"FAIL");
  return cmp;
}


TEST(test_ckwreduce, RS3_FFT1) {
  std::vector<bool> results = {
      ReduceTest<float>(0, {1,1024,3}, REDUCTION_OPS::SUM, 1, {0, 0, 1}),   // RS3 FFT
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwreduce, RS3_FFT2) {
  std::vector<bool> results = {
      ReduceTest<float>(0, {1,1024,64}, REDUCTION_OPS::SUM, 1, {0, 0, 1}),  // RS3 FFT
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwreduce, RS4_TTTF1) {
  std::vector<bool> results = {
      ReduceTest<float>(3, {2,2,16,512}, REDUCTION_OPS::SUM, 1, {1, 1, 1, 0}),   // RS4 TTTF
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwreduce, RS4_TTTF2) {
  std::vector<bool> results = {
      ReduceTest<float>(4, {2,2,2,16}, REDUCTION_OPS::SUM, 1, {1, 1, 1, 0}),     // RS4 TTTF
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwreduce, RM4_FFTF1) {
  std::vector<bool> results = {
      ReduceTest<float>(0, {2,2,3,32}, REDUCTION_OPS::MAX, 1, {0, 0, 1, 0}),   // RM4 FFTF
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwreduce, RM4_FFTF2) {
  std::vector<bool> results = {
      ReduceTest<float>(0, {2,2,3,16}, REDUCTION_OPS::MAX, 1, {0, 0, 1, 0}),   // RM4 FFTF
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwreduce, RM4_FFTF3) {
  std::vector<bool> results = {
      ReduceTest<float>(0, {2,1,1,1024}, REDUCTION_OPS::MAX, 1, {0, 0, 1, 0}), // RM4 FFTF
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
TEST(test_ckwreduce, RM4_FTFF1) {
  std::vector<bool> results = {
      ReduceTest<float>(0, {2,2,1,1024}, REDUCTION_OPS::MAX, 1, {0, 1, 0, 0}), // RM4 FTFF
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
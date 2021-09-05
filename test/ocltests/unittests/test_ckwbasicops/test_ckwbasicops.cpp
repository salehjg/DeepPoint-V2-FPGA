#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"
#include <vector>

template <typename T>
bool BasicOpsTestNonScalar(const std::vector<unsigned> &shape1, const std::vector<unsigned> &shape2){
  auto srcTn1 = GenerateTensor<T>(0,shape1);
  auto srcTn2 = GenerateTensor<T>(0,shape2);
  std::vector<bool> results;
  vector<BASIC_OPS> ops = {BASIC_OPS::ADD, BASIC_OPS::SUB, BASIC_OPS::MUL_ELEMENTWISE, BASIC_OPS::DIV_ELEMENTWISE};
  for(BASIC_OPS op:ops){
    auto goldTn = platSelection->BasicOps(PLATFORMS::CPU, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2),op);
    auto dstTn = platSelection->BasicOps(PLATFORMS::XIL, Convert2TnBasePtr(srcTn1), Convert2TnBasePtr(srcTn2),op);
    bool cmp = platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
    if(!cmp)SPDLOG_LOGGER_TRACE(logger, "BasicOps, Rank{} and Rank{}: {}", srcTn1->GetRank(),srcTn2->GetRank(), cmp?"PASS":"FAIL");
    results.push_back(cmp);
  }
  bool result = true;
  for(bool r:results){result = result && r;}
  return result;
}
template <typename T>
bool BasicOpsTestScalar(const std::vector<unsigned> &shape1, const float scalar){
  auto srcTn1 = GenerateTensor<T>(0,shape1);
  std::vector<bool> results;
  vector<BASIC_OPS> ops = {BASIC_OPS::ADD, BASIC_OPS::SUB, BASIC_OPS::MUL_ELEMENTWISE, BASIC_OPS::DIV_ELEMENTWISE};
  for(BASIC_OPS op:ops){
    auto goldTn = platSelection->BasicOps(PLATFORMS::CPU, Convert2TnBasePtr(srcTn1), scalar,op);
    auto dstTn = platSelection->BasicOps(PLATFORMS::XIL, Convert2TnBasePtr(srcTn1), scalar,op);
    bool cmp = platSelection->CompareTensors(PLATFORMS::CPU, goldTn, dstTn);
    if(!cmp)SPDLOG_LOGGER_TRACE(logger, "BasicOps, Rank{} and Scalar: {}", srcTn1->GetRank(),cmp?"PASS":"FAIL");
    results.push_back(cmp);
  }
  bool result = true;
  for(bool r:results){result = result && r;}
  return result;
}

TEST(test_ckwbasicops, mixed1) {
  std::vector<bool> results = {
      BasicOpsTestNonScalar<float>({2,2,2,2},{2,2,2,2}),
      BasicOpsTestNonScalar<float>({2,2,2,2},{2}),
      BasicOpsTestNonScalar<float>({2,2,2,17},{17}),
      BasicOpsTestScalar<float>({2,2,2,17},1.5f),

      BasicOpsTestNonScalar<float>({2,2,2},{2,2,2}),
      BasicOpsTestNonScalar<float>({2,2,2},{2}),
      BasicOpsTestNonScalar<float>({2,2,17},{17}),
      BasicOpsTestScalar<float>({2,2,17},1.5f),

      BasicOpsTestNonScalar<float>({2,2},{2,2}),
      BasicOpsTestNonScalar<float>({2,2},{2}),
      BasicOpsTestNonScalar<float>({2,17},{17}),
      BasicOpsTestScalar<float>({2,17},1.5f),

      BasicOpsTestNonScalar<float>({15},{15}),
      BasicOpsTestScalar<float>({17},1.5f),
  };

  for(auto r:results){
    EXPECT_TRUE(r);
  }
}
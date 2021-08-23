#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "fpga/xilinx/CTensorXil.h"
#include "test_helpers.h"

TEST(test_ctensor, subtest1) {
  const unsigned N=1024;
  CTensor<int> tn1({N});
  for(int i=0;i<tn1.GetLen();i++){tn1[i]=i;}

}
#include "gtest/gtest.h"
#include "cpu/CTensor.h"
#include "test_helpers.h"

TEST(test_ctensor, subtest1) {
  const unsigned N=1024;
  CTensor<unsigned> tn({N});
  EXPECT_FALSE (tn.IsEmpty());
  EXPECT_EQ (tn.GetLen(),  N);
  EXPECT_EQ (tn.GetSizeBytes(),  N*sizeof(unsigned));
  EXPECT_EQ (tn.GetRank(),  1);

  for(int i=0; i<tn.GetLen(); i++){
    tn[i]=i;
  }

}

TEST(test_ctensor, subtest2) {
  const unsigned N=1024;
  CTensor<unsigned> tn1({N,N});

  for(int i=0; i<tn1.GetLen(); i++){
    tn1[i]=i;
  }

  CTensor<unsigned> tn2(tn1);

  EXPECT_EQ (tn1.GetLen(),  tn2.GetLen());
  EXPECT_EQ (tn1.GetRank(),  tn2.GetRank());
  EXPECT_EQ (tn1.GetSizeBytes(),  tn2.GetSizeBytes());

  bool cmp = platSelection->CompareTensors(PLATFORMS::CPU, (CTensorBase*)(&tn1), (CTensorBase*)(&tn2));
  EXPECT_TRUE(cmp);
}

TEST(test_ctensor, subtest3) {
  const unsigned N=1024;
  CTensor<float> tn1({N,N});

  for(int i=0; i<tn1.GetLen(); i++){
    tn1[i]=i*0.25f;
  }

  CTensor<float> tn2(tn1);

  EXPECT_EQ (tn1.GetLen(),  tn2.GetLen());
  EXPECT_EQ (tn1.GetRank(),  tn2.GetRank());
  EXPECT_EQ (tn1.GetSizeBytes(),  tn2.GetSizeBytes());

  bool cmp = platSelection->CompareTensors(PLATFORMS::CPU, (CTensorBase*)(&tn1), (CTensorBase*)(&tn2));
  EXPECT_TRUE(cmp);
}

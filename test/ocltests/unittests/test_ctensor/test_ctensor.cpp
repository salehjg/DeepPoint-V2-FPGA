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
  CTensorPtr<unsigned> tn1(new CTensor<unsigned>({N,N}));

  for(int i=0; i<tn1->GetLen(); i++){
    (*(tn1))[i]=i;
  }

  CTensorPtr<unsigned> tn2(new CTensor<unsigned>(*tn1));

  EXPECT_EQ (tn1->GetLen(),  tn2->GetLen());
  EXPECT_EQ (tn1->GetRank(),  tn2->GetRank());
  EXPECT_EQ (tn1->GetSizeBytes(),  tn2->GetSizeBytes());

  bool cmp = platSelection->CompareTensors(
      PLATFORMS::CPU,
      Convert2TnBasePtr(tn1),
      Convert2TnBasePtr(tn2)
  );
  EXPECT_TRUE(cmp);
}

TEST(test_ctensor, subtest3) {
  const unsigned N=1024;
  CTensorPtr<float> tn1(new CTensor<float>({N,N}));

  for(int i=0; i<tn1->GetLen(); i++){
    (*tn1)[i]=i*0.25f;
  }

  CTensorPtr<float> tn2(new CTensor<float>(*tn1));

  EXPECT_EQ (tn1->GetLen(),  tn2->GetLen());
  EXPECT_EQ (tn1->GetRank(),  tn2->GetRank());
  EXPECT_EQ (tn1->GetSizeBytes(),  tn2->GetSizeBytes());

  bool cmp = platSelection->CompareTensors(
      PLATFORMS::CPU,
      Convert2TnBasePtr(tn1),
      Convert2TnBasePtr(tn2)
  );
  EXPECT_TRUE(cmp);
}

TEST(test_ctensor, expandsqueeze1) {
  CTensor<float> tn1({1,2,3,4,1});
  EXPECT_EQ(5, tn1.GetRank());
  EXPECT_EQ(1,  tn1.SqueezeDimZeroToRankTry(2));
  EXPECT_EQ(0,  tn1.SqueezeDimZeroToRankTry(1));
  EXPECT_EQ(0,  tn1.SqueezeDimZeroToRankTry(0));
  tn1.SqueezeDims();
  EXPECT_EQ(3,  tn1.GetRank());
  EXPECT_EQ(2,  tn1.ExpandDimZeroToRank(5));
  EXPECT_EQ(1,  tn1.GetShape()[0]);
  EXPECT_EQ(1,  tn1.GetShape()[1]);
  EXPECT_EQ(5,  tn1.GetRank());
}

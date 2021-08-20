#include "gtest/gtest.h"
#include "cpu/CTensor.h"

TEST(test_ctensor, subtest1) {
  unsigned N=1024;
  CTensor<int> tn({N});
  EXPECT_FALSE (tn.IsEmpty());
  EXPECT_EQ (tn.GetLen(),  N);
  EXPECT_EQ (tn.GetSizeBytes(),  N*sizeof(int));
  EXPECT_EQ (tn.GetRank(),  1);

  for(int i=0; i<tn.GetLen(); i++){
    tn[i]=i;
  }

}

TEST(test_ctensor, subtest2) {
  unsigned N=1024;
  CTensor<int> tn1({N,N});

  for(int i=0; i<tn1.GetLen(); i++){
    tn1[i]=i;
  }

  CTensor<int> tn2(tn1);

  EXPECT_EQ (tn1.GetLen(),  tn2.GetLen());
  EXPECT_EQ (tn1.GetRank(),  tn2.GetRank());
  EXPECT_EQ (tn1.GetSizeBytes(),  tn2.GetSizeBytes());

  bool cmp=true;
  for(int i=0; i<tn1.GetLen(); i++){
    if(tn1[i]!=tn2[i]){cmp=false;break;}
  }
  EXPECT_TRUE(cmp);
}

TEST(test_ctensor, subtest3) {
  unsigned N=1024;
  CTensor<float> tn1({N,N});

  for(int i=0; i<tn1.GetLen(); i++){
    tn1[i]=i*0.25f;
  }

  CTensor<float> tn2(tn1);

  EXPECT_EQ (tn1.GetLen(),  tn2.GetLen());
  EXPECT_EQ (tn1.GetRank(),  tn2.GetRank());
  EXPECT_EQ (tn1.GetSizeBytes(),  tn2.GetSizeBytes());

  bool cmp=true;
  for(int i=0; i<tn1.GetLen(); i++){
    if(tn1[i]!=tn2[i]){cmp=false;break;}
  }
  EXPECT_TRUE(cmp);
}

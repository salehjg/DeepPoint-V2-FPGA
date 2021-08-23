#pragma once

#include "cpu/CTensor.h"
#include <iostream>

using namespace std;

template <typename T>
bool CompareCTensors(CTensor<T> &tn1, CTensor<T> &tn2){
  if(tn1.GetShape()!=tn2.GetShape()) return false;
  bool matches = true;
  for(size_t i=0; i<tn1.GetLen(); i++){
    if(tn1[i]!=tn2[i]){
      cout<<__func__<<": Mismatch at index="<<i<<"tn1[index]="<<tn1[i]<<",tn2[index]="<<tn2[i]<<endl;
      matches = false
    }
  }
  return matches;
}
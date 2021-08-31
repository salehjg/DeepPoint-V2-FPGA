#pragma once
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "cpu/CTensor.h"
#include "CPlatformSelection.h"
#include <iostream>

using namespace std;

extern CPlatformSelection *platSelection;

template <typename T>
bool CompareCTensors(CTensor<T> &tn1, CTensor<T> &tn2){
  if(tn1.GetShape()!=tn2.GetShape()) return false;
  bool matches = true;
  for(size_t i=0; i<tn1.GetLen(); i++){
    if(tn1[i]!=tn2[i]){
      SPDLOG_LOGGER_ERROR(logger, "CompareCTensors: Mismatch at tn1[{}]={}, tn2[{}]={}",i,tn1[i],i,tn2[i]);
      matches = false;
    }
  }
  return matches;
}

inline float float_rand( float min, float max )
{
  float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
  return min + scale * ( max - min );      /* [min, max] */
}

template <typename T>
CTensor<T>* GenerateTensor(int pattern, const std::vector<unsigned> &shape){
  auto *testTn = new CTensor<T>(shape);
  size_t _len = testTn->GetLen();
  T *buff = testTn->Get();
  if(pattern==-1){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = 0;
    }
  }
  if(pattern==0){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = (T)float_rand(0,2.50f);
    }
  }
  if(pattern==1){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = (T) (i %5) ;//+ float_rand(0,2.50f);
    }
  }
  if(pattern==2){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = (T) (i %10 + float_rand(0,2.50f));
    }
  }
  if(pattern==3){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = (T) pattern;
    }
  }
  if(pattern==4){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = (T) pattern;
    }
  }
  if(pattern==5){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = (T) pattern;
    }
  }
  if(pattern==6){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = (T) i;
    }
  }
  if(pattern==7){
    for (size_t i = 0; i < _len; i++) {
      buff[i] = (T) float_rand(-2.50f,2.50f);
    }
  }
  return testTn;
}
#pragma once

#define DO_PRAGMA(x) _Pragma ( #x )

constexpr unsigned ConstexperDivCeil(unsigned a, unsigned b){
  return  (a-1)/b +1;
}

template<typename T>
inline T DivCeil(T a, T b){
#pragma HLS INLINE
  return ((T)(a-1)/(T)b)+1;
}

template<typename T>
inline T MakeDivisible(T value, T by){
#pragma HLS INLINE
  return (value%by==0)?
         value:
         value+(by-value%by);
}

constexpr unsigned ConstexperFloorLog2(unsigned x)
{
  return x == 1 ? 0 : 1+ConstexperFloorLog2(x >> 1);
}

constexpr unsigned ConstexperCeilLog2(unsigned x)
{
  return x == 1 ? 0 : ConstexperFloorLog2(x - 1) + 1;
}

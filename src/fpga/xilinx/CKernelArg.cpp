#include "fpga/xilinx/CKernelArg.h"
#include "CStringFormatter.h"

template<typename T>
CKernelArg<T>::CKernelArg(T *argList) {
  storedArgPtr = argList;
}

template<typename T>
T *CKernelArg<T>::Get() {
  if(storedArgPtr== nullptr)
    throw std::runtime_error(CStringFormatter()<<__func__<<": Empty storedArgPtr!");
  return storedArgPtr;
}

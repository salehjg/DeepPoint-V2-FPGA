#pragma once

class CKernelArgBase {

};

template <typename T>
class CKernelArg: public CKernelArgBase {
 public:
  CKernelArg(T *argList);
  T* Get();
 private:
  T *storedArgPtr = nullptr;
};





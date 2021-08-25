#include <iostream>
#include <vector>
#include <typeinfo>
#include "cpu/CTensor.h"
#include "fpga/xilinx/CTensorXil.h"

using namespace std;


/*
class TensorBase {
 public:
  void expandDimZero() {
    this->shape.push_back(1);
  }
  bool IsEmpty() const{
    return shape.empty();
  }

  virtual const type_info& getType() = 0;

 protected:
  vector<int> shape;
};

template<typename T>
class Tensor : public TensorBase {
 public:
  Tensor() {

  }
  Tensor(const Tensor *other) {
    bool val = other->IsEmpty();
  }
  Tensor(const vector<int> &shape) {
    this->shape = shape;
    data = new T[100];
  }

  const type_info& getType() override {
    return typeid(T);
  }

 private:
  T *data;
};

template<typename T>
class TensorFPGA : public Tensor<T> {
 public:
  TensorFPGA(const vector<int> &shape){
    this->shape = shape;
    data = new T[100];
  }

  const type_info& getType() override {
    return typeid(T);
  }

  int randomFunc1(){
    return 1;
  }

 private:
  T *data;
};

void Compute(TensorBase *tn1, TensorBase *tn2) {
  std::cout << "tn1::typeIsFloat " << (tn1->getType()== typeid(float)) << std::endl;
  std::cout << "tn1::typeIsInt " << (tn1->getType()== typeid(int))  << std::endl;
}


int main() {

  Tensor<float> *tn1;
  Tensor<int> *tn2;
  tn1 = new Tensor<float>({20});
  tn2 = new TensorFPGA<int>({100});

  Compute(tn1, tn2);

  return 0;
}
 */

int main(){
  CTensor<int> tn({200});
  int len = tn.GetLen();
  int size = tn.GetSizeBytes();
  int rank = tn.GetRank();
}
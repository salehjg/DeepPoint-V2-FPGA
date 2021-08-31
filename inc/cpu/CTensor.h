#pragma once
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE


#include "CTensorBase.h"
#include <memory>
#include "CStringFormatter.h"
#include <typeinfo>
#include <algorithm>


template <typename T>
class CTensor: public CTensorBase {
 public:
  using Ptr = std::shared_ptr<CTensor>;
  CTensor(const CTensor<T>& other);
  CTensor(const std::vector<unsigned> &shape);
  CTensor(const std::vector<unsigned> &shape, T* srcBuffToBeCopied);
  virtual CTensor& operator=(const CTensor<T>& other);
  unsigned long GetSizeBytes() const override;
  virtual const std::type_info& GetType() const;
  virtual T& operator[](std::size_t flattenedRowMajorIndex);
  virtual T* Get();
  virtual const T* GetConst() const;

 protected:
  unsigned long CheckShape(const std::vector<unsigned> &shape);
 private:
  void CloneFrom(const CTensor<T> &other);

  //using BuffType = std::unique_ptr<T[], decltype(&free)>;
  using BuffType = std::unique_ptr<T[]>;

  BuffType m_pHostBuffAligned;
};

template<typename T>
CTensor<T>::CTensor(const CTensor<T> &other) {
  CloneFrom(other);
}

template<typename T>
CTensor<T> &CTensor<T>::operator=(const CTensor<T> &other) {
  // this = other !
  CloneFrom(other);
  return *this;
}

template<typename T>
unsigned long CTensor<T>::GetSizeBytes() const {
  return GetLen()*sizeof(T);
}

template<typename T>
const std::type_info &CTensor<T>::GetType() const{
  return typeid(T);
}

template<typename T>
void CTensor<T>::CloneFrom(const CTensor<T> &other) {
  if(other.IsEmpty())
    throw std::runtime_error(CStringFormatter()<< __func__ << ": The given instance is empty!");
  if(other.GetType()!=GetType())
    throw std::runtime_error(CStringFormatter()<< __func__ << ": The given instance has a different type!");
  if (this != &other){ // not a self-assignment
    SetPlatform(PLATFORMS::CPU);
    if(other.GetLen()!=GetLen()){ // reuse already available buffer of the same size
      SetShape(other.GetShape());
      void *ptr = nullptr;
      if (posix_memalign(&ptr, 4096, other.GetLen() * sizeof(T)))
        throw std::runtime_error(
            CStringFormatter() << __func__ << ": Failed to allocate aligned memory (std::bad_alloc())!");
      m_pHostBuffAligned.reset(reinterpret_cast<T *>(ptr));
    }
    std::copy(&other.m_pHostBuffAligned[0], &other.m_pHostBuffAligned[0] + GetLen(), &m_pHostBuffAligned[0]);
  }
}

template<typename T>
T &CTensor<T>::operator[](size_t flattenedRowMajorIndex) {
  return m_pHostBuffAligned[flattenedRowMajorIndex];
}

template<typename T>
CTensor<T>::CTensor(const std::vector<unsigned> &shape) {
  SetPlatform(PLATFORMS::CPU);
  const unsigned long newLen = accumulate(begin(shape), end(shape), 1, std::multiplies<unsigned>());
  if(newLen<1)
    throw std::runtime_error(CStringFormatter()<< __func__ << ": Bad tensor shape to create a tensor with.");
  SetShape(shape);
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 4096, newLen * sizeof(T)))
    throw std::runtime_error(CStringFormatter() << __func__ << ": Failed to allocate aligned memory (bad_alloc())!");
  m_pHostBuffAligned.reset(reinterpret_cast<T *>(ptr));
}

template<typename T>
CTensor<T>::CTensor(const std::vector<unsigned> &shape, T *srcBuffToBeCopied) {
  SetPlatform(PLATFORMS::CPU);
  const unsigned long newLen = CheckShape(shape);
  SetShape(shape);
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 4096, newLen * sizeof(T)))
    throw std::runtime_error(CStringFormatter() << __func__ << ": Failed to allocate aligned memory (bad_alloc())!");
  m_pHostBuffAligned.reset(reinterpret_cast<T *>(ptr));
  std::copy(&srcBuffToBeCopied[0], &srcBuffToBeCopied[0] + newLen, &m_pHostBuffAligned[0]);
}

template<typename T>
unsigned long CTensor<T>::CheckShape(const std::vector<unsigned> &shape) {
  const unsigned long newLen = accumulate(begin(shape), end(shape), 1, std::multiplies<unsigned>());
  if(newLen<1)
    throw std::runtime_error(CStringFormatter()<< __func__ << ": Bad tensor shape to create a tensor with.");
  return newLen;
}
template<typename T>
T *CTensor<T>::Get() {
  return m_pHostBuffAligned.get();
}
template<typename T>
const T *CTensor<T>::GetConst() const {
  return m_pHostBuffAligned.get();
}
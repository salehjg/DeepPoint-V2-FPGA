#pragma once

#include "CTensorBase.h"
#include <memory>
#include "CStringFormatter.h"
#include <typeinfo>
#include <algorithm>


template <typename T>
class CTensorXil: public CTensor {
 public:
  using Ptr = std::shared_ptr<CTensor>;
  CTensor(const CTensor<T>& other);
  CTensor(const std::vector<unsigned> &shape);
  CTensor(const std::vector<unsigned> &shape, T* srcBuffToBeCopied);
  CTensor& operator=(const CTensor<T>& other);
  unsigned long GetSizeBytes() const override;
  virtual const std::type_info& GetType() const;
  virtual T& operator[](std::size_t flattenedRowMajorIndex);

 protected:

 private:
  void CloneFrom(const CTensor<T> &other);
  unsigned long CheckShape(const std::vector<unsigned> &shape);
  //using BuffType = std::unique_ptr<T[], decltype(&free)>;
  using BuffType = std::unique_ptr<T[]>;

  BuffType m_pHostBuffAligned;
};

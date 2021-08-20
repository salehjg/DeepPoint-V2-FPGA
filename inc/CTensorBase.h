#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
#include <algorithm>

class CTensorBase {
 public:
  const std::vector<unsigned>& GetShape() const;
  unsigned GetRank() const;
  void SqueezeDims();
  void ExpandDims(const unsigned axis);
  void SqueezeDimZero();
  void ExpandDimZero();
  void Reshape(const std::vector<unsigned>& newShape);
  unsigned long GetLen() const;
  virtual unsigned long GetSizeBytes() const = 0;
  bool IsEmpty() const;


 protected:
  void SetShape(const std::vector<unsigned> &newShape);
 private:
  std::vector<unsigned> m_vShape;
};
#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "GlobalHelpers.h"

class CTensorBase {
 public:
  const std::vector<unsigned> GetShape() const;
  unsigned GetRank() const;
  void SqueezeDims();
  void ExpandDims(const unsigned axis);
  void SqueezeDimZero();
  void ExpandDimZero();
  void Reshape(const std::vector<unsigned>& newShape);
  unsigned long GetLen() const;
  virtual unsigned long GetSizeBytes() const = 0;
  bool IsEmpty() const;
  PLATFORMS GetPlatform();

 protected:
  void SetShape(const std::vector<unsigned> &newShape);
  void SetPlatform(PLATFORMS platform);
 private:
  std::vector<unsigned> m_vShape;
  PLATFORMS m_ePlatform;
};
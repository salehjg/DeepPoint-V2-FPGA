#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "GlobalHelpers.h"

class CTensorBase {
 public:
  /**
   * Returns a copy of the tensor's shape.
   * @return
   */
  const std::vector<unsigned> GetShape() const;

  /**
   * Returns the tensor's rank.
   * @return
   */
  unsigned GetRank() const;

  /**
   * Removes all the elements in the tensor's shape which are equal to 1.
   */
  void SqueezeDims();

  /**
   * Inserts an item of value 1 to the tensor's shape at the given axis.
   * @param axis should be    -GetRank()-1 <= axis <= GetRank()
   */
  void ExpandDims(const unsigned axis);

  /**
   * Removes the first item (index 0) of the tensor's shape if its value is 1.
   * @return returns true if the tensor's rank has been changed.
   */
  bool SqueezeDimZero();

  /**
   * Inserts an item of value 1 to the index 0 of the tensor's shape.
   */
  void ExpandDimZero();

  /**
   * Keeps calling ExpandDimZero() until the tensor's rank reaches targetRank
   * @param targetRank
   * @return returns the total number of items of value 1 that are inserted.
   */
  unsigned ExpandDimZeroToRank(unsigned targetRank);

  /**
   * Calls SqueezeDimZero `targetRank` times.
   * @param targetRank
   * @return returns the total number of items of value 1 that are removed.
   */
  unsigned SqueezeDimZeroToRankTry(unsigned targetRank);

  /**
   * Replaces the shape of the tensor if they are both of the same length.
   * Please note that Reshape() does not affect the data layout in anyways.
   * @param newShape
   */
  void Reshape(const std::vector<unsigned>& newShape);

  /**
   * Returns the total number of items stored in the tensor.
   * @return
   */
  unsigned long GetLen() const;

  /**
   * Returns the total size of the tensor in bytes.
   * @return
   */
  virtual unsigned long GetSizeBytes() const = 0;

  /**
   * returns true if the tensor's shape vector is empty.
   * @return
   */
  bool IsEmpty() const;

  /**
   * Returns the platform that the tensor object is on.
   * @return
   */
  PLATFORMS GetPlatform();

  /**
   * Returns true if the tensor type is float 32.
   * @return
   */
  bool IsTypeFloat32() const;


  /**
   * Returns true if the tensor type is unsigned int 32.
   * @return
   */
  bool IsTypeUint32() const;


  /**
   * Returns true if the tensor type is int 32.
   * @return
   */
  bool IsTypeInt32() const;

 protected:
  CTensorBase();
  void SetShape(const std::vector<unsigned> &newShape);
  void SetPlatform(PLATFORMS platform);
  bool m_bTypeIsFloat;
  bool m_bTypeIsUint;
  bool m_bTypeIsInt;

 private:
  std::vector<unsigned> m_vShape;
  PLATFORMS m_ePlatform;
};

using CTensorBasePtr = std::shared_ptr<CTensorBase>;

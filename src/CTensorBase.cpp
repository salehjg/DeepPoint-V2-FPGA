#include <CStringFormatter.h>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "CTensorBase.h"

using namespace std;

const vector<unsigned> CTensorBase::GetShape() const {
  if (m_vShape.empty()) {
    throw runtime_error(CStringFormatter() << "m_vShape is empty, is the tensor initialized?");
  }
  return m_vShape;
}

void CTensorBase::SqueezeDims() {
  if (m_vShape.empty()) {
    throw runtime_error(CStringFormatter() << "m_vShape is empty, is the tensor initialized?");
  }
  m_vShape.erase(std::remove(m_vShape.begin(), m_vShape.end(), 1), m_vShape.end());
}

bool CTensorBase::SqueezeDimZero() {
  if (m_vShape.empty()) {
    throw runtime_error(CStringFormatter() << "m_vShape is empty, is the tensor initialized?");
  }
  if (m_vShape[0] == 1) {
    m_vShape.erase(m_vShape.begin());
    return true;
  }else{
    return false;
  }
}

void CTensorBase::ExpandDims(const unsigned axis) {
  if (m_vShape.empty()) {
    throw runtime_error(CStringFormatter() << "m_vShape is empty, is the tensor initialized?");
  }
  if (axis >= 0) {
    if (axis > GetRank())
      throw runtime_error(CStringFormatter() << __func__ << ": axis>=0 should not be bigger than GetRank().");
  } else {
    if (axis < -1 * GetRank() - 1)
      throw runtime_error(CStringFormatter() << __func__ << ": axis<0 should not be smaller than -1*GetRank()-1.");
  }
  m_vShape.insert(m_vShape.begin() + ((axis < 0) ? (axis + GetRank()) : axis), 1);
}

void CTensorBase::ExpandDimZero() {
  ExpandDims(0);
}

void CTensorBase::Reshape(const vector<unsigned> &newShape) {
  if (newShape.empty()) {
    throw runtime_error(CStringFormatter() << "The new shape is empty.");
  }
  const unsigned long newLen = std::accumulate(begin(newShape), end(newShape), 1, multiplies<unsigned>());
  if (newLen != GetLen())
    throw runtime_error(CStringFormatter() << __func__ << ": The lengths for the two shapes do not match.");
  m_vShape = newShape;
}

unsigned long CTensorBase::GetLen() const{
  if (IsEmpty()) {
    return 0;
  }
  return std::accumulate(begin(m_vShape), end(m_vShape), 1, multiplies<unsigned>());
}

unsigned CTensorBase::GetRank() const{
  if (m_vShape.empty()) {
    throw runtime_error(CStringFormatter() << "m_vShape is empty, is the tensor initialized?");
  }
  return m_vShape.size();
}

void CTensorBase::SetShape(const std::vector<unsigned> &newShape) {
  m_vShape = newShape;
}

bool CTensorBase::IsEmpty() const{
  return m_vShape.empty();
}
PLATFORMS CTensorBase::GetPlatform() {
  return m_ePlatform;
}
void CTensorBase::SetPlatform(PLATFORMS platform) {
  m_ePlatform = platform;
}
bool CTensorBase::IsTypeFloat32() const {
  return m_bTypeIsFloat;
}
bool CTensorBase::IsTypeUint32() const {
  return m_bTypeIsUint;
}
bool CTensorBase::IsTypeInt32() const {
  return m_bTypeIsInt;
}
CTensorBase::CTensorBase() {
}
unsigned CTensorBase::ExpandDimZeroToRank(unsigned targetRank) {
  if(targetRank <= GetRank())
    return 0;
  else {
    const unsigned diff = targetRank - GetRank();
    unsigned i = diff;
    while(i--){
      ExpandDimZero();
    }
    return diff;
  }
}
unsigned CTensorBase::SqueezeDimZeroToRankTry(unsigned targetRank) {
  if(GetRank()<= targetRank)
    return 0;
  else{
    unsigned diff = 0;
    unsigned i=GetRank() - targetRank;
    while(i--){
      diff += SqueezeDimZero() ? 1 : 0;
    }
    return diff;
  }

}


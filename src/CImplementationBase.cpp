#include "CImplementationBase.h"
unsigned CImplementationBase::GenerateLayerId() {
  m_uAtomicCounter++;
}
PLATFORMS CImplementationBase::GetPlatform() const{
  return m_ePlatform;
}
void CImplementationBase::ResetLayerIdCounter(unsigned offset) {
  m_uAtomicCounter.store(offset);
}
unsigned CImplementationBase::GetTheLastLayerId() {
  return m_uAtomicCounter;
}

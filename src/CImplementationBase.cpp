#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

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
void CImplementationBase::ValidateTensorPlatforms(const std::vector<CTensorBase *> &tensors, PLATFORMS requiredPlatform) {
  for(CTensorBase * tn:tensors){
    if(tn->GetPlatform()!=requiredPlatform){
      throw std::runtime_error(CStringFormatter() << __func__ << ": The input tensors are on the wrong platform!");
    }
  }
}

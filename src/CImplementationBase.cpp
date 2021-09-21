#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "CImplementationBase.h"
unsigned CImplementationBase::GenerateLayerId() {
  return m_uAtomicCounter++;
}
PLATFORMS CImplementationBase::GetPlatform() const{
  return m_ePlatform;
}
void CImplementationBase::ResetLayerIdCounter(unsigned offset) {
  m_uAtomicCounter.store(offset);
}
unsigned CImplementationBase::GetTheLastLayerId() {
  return m_uAtomicCounter -1;
}
void CImplementationBase::ValidateTensorPlatforms(const std::vector<CTensorBasePtr> &tensors, PLATFORMS requiredPlatform) {
  for(auto &tn:tensors){
    if(tn->GetPlatform()!=requiredPlatform){
      ThrowException("The input tensors are on the wrong platform!");
    }
  }
}

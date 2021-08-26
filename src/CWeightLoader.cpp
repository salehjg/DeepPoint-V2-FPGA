#include "CWeightLoader.h"
CWeightLoader::CWeightLoader() {

}
void CWeightLoader::LoadWeightsFromDisk(std::string &weightsBaseDir,
                                        std::string &pathToTxtFnameList,
                                        CXilinxInfo *xilInfo) {

}
CTensorBase *CWeightLoader::AccessWeights(PLATFORMS &platform, std::string &name) {
  return nullptr;
}
int CWeightLoader::ResolveMemoryBank(PLATFORMS &platform, std::string &name) {
  return 0;
}
int CWeightLoader::_ResolveMemoryBankOclXilinx(std::string &name) {
  return 0;
}
std::string CWeightLoader::_ResolveTensorTagOclXilinx(std::string &name) {
  return std::string();
}

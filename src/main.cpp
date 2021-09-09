#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "GlobalHelpers.h"
#include "CClassifierMultiPlatform.h"

int main(int argc, const char* argv[]){
  SetupModules(argc,argv);
  CClassifierMultiPlatform classifier(
      globalShapenet,
      globalProfileOclEnabled,
      globalDumpMemBankCrossings,
      globalCpuUsageSamplingEnabled,
      globalDumpTensors);
}

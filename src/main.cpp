#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "GlobalHelpers.h"
#include "CClassifierMultiPlatform.h"

CClassifierMultiPlatform *classifier;

int main(int argc, const char* argv[]){
  SetupModules(argc,argv);
  classifier = new CClassifierMultiPlatform(
      globalShapenet,
      globalProfileOclEnabled,
      globalDumpMemBankCrossings,
      globalCpuUsageSamplingEnabled,
      globalDumpTensors);
  SPDLOG_LOGGER_TRACE(logger, "The forward pass has finished.");
  delete(classifier);
  SPDLOG_LOGGER_TRACE(logger, "Closing.");
}

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <string>
#include "gtest/gtest.h"
#include "GlobalHelpers.h"
#include "CPlatformSelection.h"

using namespace std;
CPlatformSelection *platSelection;

int main(int argc, char** argv){
  SetupModules(argc, (const char**)argv);
  platSelection = new CPlatformSelection(
      false,
      globalProfileOclEnabled,
      globalDumpMemBankCrossings,
      false,
      globalDumpTensors
  );
  ::testing::InitGoogleTest(&argc, argv);
  auto exitCode = RUN_ALL_TESTS();

  delete platSelection;
  return exitCode;
}

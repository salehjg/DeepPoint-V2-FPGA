#pragma once

#include <string>
#include "CStringFormatter.h"
#include <iostream>
#include <execinfo.h>
#include <unistd.h>
#include <csignal>
#include "fpga/xilinx/xcl2.h"
#include "spdlog/spdlog.h"

//https://github.com/gabime/spdlog/wiki/0.-FAQ#how-to-remove-all-debug-statements-at-compile-time-
//#undef SPDLOG_ACTIVE_LEVEL
//#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

enum class PLATFORMS{
  CPU,
  XIL
};

enum class BASIC_OPS{
  ADD,
  SUB,
  MUL_ELEMENTWISE,
  DIV_ELEMENTWISE
};

enum class REDUCTION_OPS{
  SUM,
  MAX
};

struct CallbackData{
  void *classPtr;
  unsigned parentLayerId;
  unsigned kernelBookKeeperId;
  bool profileKernel;
};

struct ProfiledLaunchData{
  unsigned parentLayerId;
  std::string taskName;
  cl_ulong durationOcl;
};

extern spdlog::logger *logger;
extern std::string globalArgXclBin;
extern std::string globalArgDataPath;

extern unsigned globalBatchsize;
extern bool globalDumpTensors;
extern bool globalDumpMemBankCrossings;
extern bool globalProfileOclEnabled;
extern bool globalCpuUsageSamplingEnabled;
extern bool globalModelnet;
extern bool globalShapenet;

extern void SetupModules(int argc, const char* argv[]);

/*
template <typename T>
inline void OclCheckOLD(cl_int status, T command){
  if(status != CL_SUCCESS){
    std::string str = CStringFormatter()<<__FILE__<<":"<<__func__<<":"<<__LINE__<<":: OCL Status="<<status;
    //SPDLOG_LOGGER_ERROR(logger,str);
    throw std::runtime_error(str);
  }
}
*/

#define OclCheck(error,call)                                       \
    call;                                                           \
    if (error != CL_SUCCESS) {                                      \
      SPDLOG_LOGGER_ERROR(logger,"Error calling {}, error code is: {}", #call, error);\
      exit(EXIT_FAILURE);                                           \
    }

#define ThrowException(arg) \
    {\
    std::string _msg = CStringFormatter()<<__FILE__<<":"<<__LINE__<<": "<<arg;\
    throw std::runtime_error(_msg); \
    SPDLOG_LOGGER_ERROR(logger,_msg);\
    exit(EXIT_FAILURE);\
    }

#define ConditionCheck(condition,msgIfFalse) \
    if(!(condition)){\
    std::string _msg = CStringFormatter()<<__FILE__<<":"<<__LINE__<<": Failed "<< #condition <<": "<<msgIfFalse;\
    throw std::runtime_error(_msg); \
    SPDLOG_LOGGER_ERROR(logger,_msg);\
    exit(EXIT_FAILURE);\
    }


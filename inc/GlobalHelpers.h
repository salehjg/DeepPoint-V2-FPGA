#pragma once

#include <string>
#include "CStringFormatter.h"
#include "fpga/xilinx/xcl2.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

//https://github.com/gabime/spdlog/wiki/0.-FAQ#how-to-remove-all-debug-statements-at-compile-time-
#undef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

enum class PLATFORMS{
  CPU,
  XIL
};

enum class MAT_OPS{
  ADD,
  SUB,
  MUL_ELEMENTWISE,
  DIV_ELEMENTWISE
};

extern spdlog::logger *logger;
extern std::string globalArgXclBin;
extern std::string globalArgDataPath;
extern unsigned globalBatchsize;
extern bool globalDumpTensors;
extern bool globalProfileOcl;
extern bool globalModelnet;
extern bool globalShapenet;

template <typename T>
inline void OclCheck(cl_int status, T command){
  if(status != CL_SUCCESS){
    std::string str = CStringFormatter()<<__FILE__<<":"<<__func__<<":"<<__LINE__<<":: OCL Status="<<status;
    //SPDLOG_LOGGER_ERROR(logger,str);
    throw std::runtime_error(str);
  }
}


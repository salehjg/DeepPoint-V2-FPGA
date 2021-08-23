#pragma once

#include <string>
#include "CStringFormatter.h"
#include "fpga/xilinx/xcl2.h"

template <typename T>
inline void OclCheck(cl_int status, T command){
  if(status != CL_SUCCESS){
    std::string str = CStringFormatter()<<__FILE__<<":"<<__func__<<":"<<__LINE__<<":: OCL Status="<<status;
    //SPDLOG_LOGGER_ERROR(logger,str);
    throw std::runtime_error(str);
  }
}
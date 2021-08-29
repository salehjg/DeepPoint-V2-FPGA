/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include "GlobalHelpers.h"
#include "fpga/xilinx/xcl2.h"
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

namespace xcl {
std::vector<cl::Device> get_devices(const std::string &vendor_name) {
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OclCheck(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++) {
        platform = platforms[i];
      std::string platformName;
        OclCheck(err,platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (platformName == vendor_name) {
            SPDLOG_LOGGER_TRACE(logger,"Found Platform");
            SPDLOG_LOGGER_TRACE(logger,"Platform Name: {}", platformName.c_str());
            break;
        }
    }
    if (i == platforms.size()) {
        SPDLOG_LOGGER_ERROR(logger,"Failed to find Xilinx platform");
        exit(EXIT_FAILURE);
    }
    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    OclCheck(err,
              err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}

std::vector<cl::Device> get_xil_devices() { return get_devices("Xilinx"); }


std::vector<unsigned char> read_binary_file(const std::string &xclbin_file_name) {
    SPDLOG_LOGGER_TRACE(logger,"Reading {}", xclbin_file_name);

    if (access(xclbin_file_name.c_str(), R_OK) != 0) {
        SPDLOG_LOGGER_ERROR(logger,"{} xclbin does not exist", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    SPDLOG_LOGGER_TRACE(logger,"Loading {}", xclbin_file_name.c_str());
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<unsigned char> buf;
    buf.resize(nb);
    bin_file.read(reinterpret_cast<char*>(buf.data()), nb);
    return buf;
}

bool is_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if (xcl_mode != NULL) {
        ret = true;
    }
    return ret;
}

bool is_hw_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if ((xcl_mode != NULL) && !strcmp(xcl_mode, "hw_emu")) {
        ret = true;
    }
    return ret;
}

bool is_xpr_device(const char *device_name) {
    const char *output = strstr(device_name, "xpr");

    if (output == NULL) {
        return false;
    } else {
        return true;
    }
}
}; // namespace xcl

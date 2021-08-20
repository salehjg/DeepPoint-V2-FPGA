cmake_minimum_required(VERSION 3.15)
project(DeepPoint_V2_FPGA)

set(CMAKE_CXX_STANDARD 14)
set(HLSLIB_ENABLE_TESTING OFF)
set(HLSLIB_BUILD_DOCUMENTATION OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/submodules/hlslib/cmake)

find_package(Threads REQUIRED)
find_package(SDAccel REQUIRED)
include(CheckTypeSize)


include_directories(
        inc/
)

add_executable(DeepPoint_V2_FPGA
        src/main.cpp
        src/CTensorBase.cpp

        )

add_subdirectory(${CMAKE_SOURCE_DIR}/submodules/googletest)
add_subdirectory(${CMAKE_SOURCE_DIR}/test)
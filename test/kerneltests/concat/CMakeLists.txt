find_package(Threads REQUIRED)
include_directories(
        ${PROJECT_SOURCE_DIR}/inc/fpga/xilinx
        ${PROJECT_SOURCE_DIR}/submodules/hlslib/include
        inc
        ${PROJECT_SOURCE_DIR}/test/kerneltests/common/inc)

add_executable(KernelTestConcat
        src/CpuTestConcat.cpp
        ${PROJECT_SOURCE_DIR}/src/fpga/xilinx/kernels/concat.cpp)

target_link_libraries(KernelTestConcat
        ${SDAccel_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${SDAccel_FLOATING_POINT_LIBRARY}
        ${SDAccel_LIBRARIES})

add_test(NAME KernelTestConcat COMMAND KernelTestConcat)
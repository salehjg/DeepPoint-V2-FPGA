include_directories(
        ${CMAKE_SOURCE_DIR}/test/ocltests/common
        ${CMAKE_SOURCE_DIR}/config/output
        ${CMAKE_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/submodules/hlslib/include
        ${CMAKE_SOURCE_DIR}/submodules/argparse
        ${CMAKE_SOURCE_DIR}/submodules/spdlog/include
        ${CMAKE_SOURCE_DIR}/submodules/rapidjson/include
        ${CMAKE_BINARY_DIR}
        ${SDAccel_INCLUDE_DIRS}
        )

file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/test/ocltests/unittests/*.cpp
        )
message("TEST_SOURCES: ")
message("${TEST_SOURCES}")

set(SOURCES
        ${CMAKE_SOURCE_DIR}/src/CTensorBase.cpp
        ${CMAKE_SOURCE_DIR}/src/CImplementationBase.cpp
        ${CMAKE_SOURCE_DIR}/src/cpu/CImplementationCpu.cpp
        ${CMAKE_SOURCE_DIR}/src/fpga/xilinx/CImplementationXilinx.cpp
        ${CMAKE_SOURCE_DIR}/src/CPlatformSelection.cpp
        ${CMAKE_SOURCE_DIR}/src/CProfiler.cpp
        ${CMAKE_SOURCE_DIR}/src/CWeightLoader.cpp
        ${CMAKE_SOURCE_DIR}/src/fpga/xilinx/CKernelWrapper.cpp
        ${CMAKE_SOURCE_DIR}/src/cnpy.cpp
        ${CMAKE_SOURCE_DIR}/src/fpga/xilinx/xcl2.cpp
        ${TEST_SOURCES}
        )
message("SOURCES: ")
message("${SOURCES}")

add_executable(OclTestsMain
        ${CMAKE_SOURCE_DIR}/test/ocltests/tests_main.cpp
        ${SOURCES}
        )

add_test(NAME OclTestsMain COMMAND OclTestsMain)
target_link_libraries(OclTestsMain gtest ${SDAccel_LIBRARIES} ${SDAccel_FLOATING_POINT_LIBRARY} z stdc++fs spdlog)
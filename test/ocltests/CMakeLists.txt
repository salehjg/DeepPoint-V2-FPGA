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

#file(GLOB_RECURSE TEST_SOURCES
#        ${CMAKE_SOURCE_DIR}/test/ocltests/unittests/*.cpp
#        )
#message("TEST_SOURCES: ")
#message("${TEST_SOURCES}")
set(TEST_SOURCES
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ctensor/test_ctensor.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ctensorxil/test_ctensorxil.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwconcat/test_ckwconcat.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwmatmul/test_ckwmatmul.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwrelusqrtsquare/test_ckwrelusqrtsquare.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwbasicops/test_ckwbasicops.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwtile/test_ckwtile.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwtranspose/test_ckwtranspose.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwgather/test_ckwgather.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwreduce/test_ckwreduce.cpp
        ${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_layermean/test_layermean.cpp
        ${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_layervariance/test_layervariance.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwpadunpad/test_ckwpadunpad.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwtopk/test_ckwtopk.cpp
        #${CMAKE_SOURCE_DIR}/test/ocltests/unittests/test_ckwconv/test_ckwconv.cpp
        )


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
        ${CMAKE_SOURCE_DIR}/src/GlobalHelpers.cpp

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
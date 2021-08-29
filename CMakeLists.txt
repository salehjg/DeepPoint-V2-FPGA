cmake_minimum_required(VERSION 3.0)
if(NOT DEFINED ENV{AWS_PLATFORM})
    message(FATAL_ERROR "You must set env variable AWS_PLATFORM to the path of the SDAccel platform file(*.xpfm)")
endif()
message("CMake Version: ${CMAKE_VERSION}")

set(HostExecutableName DeepPointV2FPGA)
project(DeepPointV2FPGA)

set(BuildRelease ON CACHE BOOL "Build the host application and CpuTests with release mode.")
set(DSA_NAME "$ENV{AWS_PLATFORM}" CACHE STRING "Known SDAccel platform name or xpfm file path")
set(SYNTH_PART_NAME "xcvu9p-flga2104-2-e" CACHE STRING "Part name for synthesis only.")
set(SYNTH_FLAGS "-DKERNEL_LOGS -DHLSLIB_SYNTHESIS -DHLSLIB_XILINX -std=c++11 -I${CMAKE_SOURCE_DIR}/inc/ocl_imp/xilinx -I${CMAKE_SOURCE_DIR}/config/output -I${CMAKE_SOURCE_DIR}/hlslib/include" CACHE STRING "CFlags for synthesis only.")
set(ENABLE_PROFILING OFF CACHE BOOL "Collect profiling information (Master Switch)")
set(PROFILE_DATA OFF CACHE BOOL "Data Profiling ON/OFF (For all of the kernels)")
set(PROFILE_STALL OFF CACHE BOOL "Stall Profiling Switch (For all of the kernels)")
set(PROFILE_EXEC OFF CACHE BOOL "Exec Profiling Switch (For all of the kernels)")
#set(OPTIMIZATION_OPTION "0" CACHE STRING "0:default, 1:reduce power, 2:increase kernel speed, 3:highest level of optimization, s:optimize for size, quick:quick compilation, This option overrides VivadoOptions.")
set(ReportLevel "0" CACHE STRING "0: no DCP, 1: all of DCPs, 2: detailed, estimate: generate design.xml")
set(KERNEL_CLOCK "130" CACHE STRING "MHz, For XOCC Compile and Link Procedures. All of the kernels will use this clock frequency. Set to -1 to ignore frequency override.")
set(UseMemoryBank0 ON CACHE BOOL "Enable/Disable usage of memory bank 0")
set(UseMemoryBank1 ON CACHE BOOL "Enable/Disable usage of memory bank 1")
set(UseMemoryBank2 OFF CACHE BOOL "Enable/Disable usage of memory bank 2")
set(UseMemoryBank3 OFF CACHE BOOL "Enable/Disable usage of memory bank 3")
set(VivadoOptions "--xp vivado_prop:run.impl_1.{STEPS.OPT_DESIGN.ARGS.DIRECTIVE}={ExploreSequentialArea} --xp vivado_prop:run.impl_1.{STEPS.PLACE_DESIGN.ARGS.DIRECTIVE}={SSI_SpreadSLLs} --xp vivado_prop:run.impl_1.{STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE}={AggressiveExplore}" CACHE STRING "Extra options for implementation steps")

set(PASTEBIN_0_1 "0" CACHE STRING "Enable or disable PasteBin agent for uploading autobuild log files to PasteBin.")
set(PASTEBIN_USERNAME "" CACHE STRING "PasteBin Account's Username")
set(PASTEBIN_PASSWORD "" CACHE STRING "PasteBin Account's Password")
set(PASTEBIN_API_KEY  "" CACHE STRING "PasteBin Account's API Key")




set(CMAKE_CXX_STANDARD 14)
set(HLSLIB_ENABLE_TESTING OFF)
set(HLSLIB_BUILD_DOCUMENTATION OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/submodules/hlslib/cmake)

find_package(Threads REQUIRED)
find_package(SDAccel REQUIRED)
include(CheckTypeSize)

include_directories(
        ${CMAKE_SOURCE_DIR}/config/output
        ${CMAKE_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/submodules/hlslib/include
        ${CMAKE_SOURCE_DIR}/submodules/argparse
        ${CMAKE_SOURCE_DIR}/submodules/spdlog/include
        ${CMAKE_BINARY_DIR}
        ${SDAccel_INCLUDE_DIRS})

add_definitions(-DREPO_DIR=\"${CMAKE_SOURCE_DIR}\")
add_definitions(-DBIN_DIR=\"${CMAKE_BINARY_DIR}\")

if(BuildRelease)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
else()
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -O0")
endif()

if(((${SDAccel_MAJOR_VERSION} LESS 2018) AND (${SDAccel_MINOR_VERSION} LESS 3)) OR ${SDAccel_MAJOR_VERSION} LESS 2017)
    add_definitions(-DHLSLIB_LEGACY_SDX=1)
else()
    add_definitions(-DHLSLIB_LEGACY_SDX=0)
endif()

include(ProcessorCount)
ProcessorCount(_CpuCount)
if(NOT ${_CpuCount} EQUAL 0)
    set(CpuCount ${_CpuCount})
else()
    set(CpuCount 4)
endif()
message("Detected CPU Core Count: ${CpuCount}")


set(MemoryBankFlags "")
if(NOT(${UseMemoryBank0} OR ${UseMemoryBank1} OR ${UseMemoryBank2} OR ${UseMemoryBank3}))
    message(FATAL_ERROR "All of the memory banks cannot be disabled.")
endif()
if(${UseMemoryBank0})
    # using lists helps to eliminate the extra space at the begining or at the end of string.
    list(APPEND MemoryBankFlags "-DUSEMEMORYBANK0")
endif()
if(${UseMemoryBank1})
    list(APPEND MemoryBankFlags "-DUSEMEMORYBANK1")
endif()
if(${UseMemoryBank2})
    list(APPEND MemoryBankFlags "-DUSEMEMORYBANK2")
endif()
if(${UseMemoryBank3})
    list(APPEND MemoryBankFlags "-DUSEMEMORYBANK3")
endif()
string(REPLACE ";" " " MemoryBankFlags "${MemoryBankFlags}" )
add_definitions(${MemoryBankFlags})
message("Kernel Flags: ${MemoryBankFlags}")

set(hw_link_command "")
set(hwemu_link_command "")
set(swemu_link_command "")
set(hw_link "")
set(hwemu_link "")
set(swemu_link "")


function(floatexpr expr output)
    set(FLOAT_EXPR_SRC "
    #include <stdio.h>
    int main(int argc, char *argv[]){printf(\"%f\", ${expr})\; return 0\;}
    ")
    FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CMakeFloatExpr.c ${FLOAT_EXPR_SRC})

    try_run(RESULT COMPRES ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/CMakeFloatExpr.c
            RUN_OUTPUT_VARIABLE __output)

    set(${output} ${__output} PARENT_SCOPE)
endfunction()


# Generates and appends commands for compiling and linking the given kernel.
function(sdaccel_target
        KERNEL_FILE
        KERNEL_FNAME_WITHOUT_EXTENSION
        KERNEL_NAME
        IS_PROFILED
        IS_MULTI_DDR_BANK
        MULTI_DDR_SP_TAG
        ADDITIONAL_FLAGS)
    set(KERNEL_FLAGS "${MemoryBankFlags} ${SYNTH_FLAGS}")

    set(XOCC_COMMAND
            -s
            -I${CMAKE_SOURCE_DIR}/inc
            -I${CMAKE_SOURCE_DIR}/hlslib/include
            -I${CMAKE_BINARY_DIR}
            --kernel ${KERNEL_NAME}
            --platform ${DSA_NAME}
            --jobs ${CpuCount}
            --xp prop:kernel.${KERNEL_NAME}.kernel_flags="${KERNEL_FLAGS}"
            --xp misc:solution_name="${KERNEL_NAME}_solution"
            ${VivadoOptions}
            )

    if(ENABLE_PROFILING AND PROFILE_STALL)
        set(XOCC_COMMAND "${XOCC_COMMAND} --profile_kernel stall:all:all")
    endif()

    if(KERNEL_CLOCK GREATER 0)
        if((${SDAccel_MAJOR_VERSION} EQUAL 2017) AND (${SDAccel_MINOR_VERSION} EQUAL 4))
            set(XOCC_COMMAND ${XOCC_COMMAND}
                    --xp prop:solution.hls_pre_tcl=KernelFreqSDx20174.tcl
                    )
        else()
            set(XOCC_COMMAND ${XOCC_COMMAND}
                    --kernel_frequency ${KERNEL_CLOCK}
                    )
        endif()
    endif()
    string (REPLACE ";" " " XOCC_COMMAND "${XOCC_COMMAND}")

    set(hw_compile "${SDAccel_XOCC} -g -c -t hw ${XOCC_COMMAND} --report_level ${ReportLevel} ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw.xo")
    set(hwemu_compile "${SDAccel_XOCC} -g -c -t hw_emu ${XOCC_COMMAND} -Restimate ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw_emu.xo")
    set(swemu_compile "${SDAccel_XOCC} -g -c -t sw_emu ${XOCC_COMMAND} ${KERNEL_FILE} -o ${KERNEL_FNAME_WITHOUT_EXTENSION}_sw_emu.xo")
    file(APPEND "${CMAKE_BINARY_DIR}/Compile_Hw_Batch.sh" "${hw_compile}\n")
    file(APPEND "${CMAKE_BINARY_DIR}/Compile_HwEmu_Batch.sh" "${hwemu_compile}\n")
    file(APPEND "${CMAKE_BINARY_DIR}/Compile_SwEmu_Batch.sh" "${swemu_compile}\n")

    #---------------------------------------------------------------------------------------------------------
    # Generating linker commands
    set(hw_link "${hw_link} ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw.xo" PARENT_SCOPE)
    set(hwemu_link "${hwemu_link} ${KERNEL_FNAME_WITHOUT_EXTENSION}_hw_emu.xo" PARENT_SCOPE)
    set(swemu_link "${swemu_link} ${KERNEL_FNAME_WITHOUT_EXTENSION}_sw_emu.xo" PARENT_SCOPE)

    set(hw_link_command "${hw_link_command} --nk ${KERNEL_NAME}:1")
    if(IS_PROFILED)
        set(hw_link_command "${hw_link_command} --profile_kernel data:${KERNEL_NAME}:all:all")
    endif()
    if(IS_MULTI_DDR_BANK)
        set(hw_link_command "${hw_link_command} ${MULTI_DDR_SP_TAG}")
    endif()

    # Identical commands for hw-emu link process
    set(hwemu_link_command "${hw_link_command}")
    set(swemu_link_command "${hw_link_command}")

    # Set link commands globally
    set(hw_link_command         "${hw_link_command}"        PARENT_SCOPE)
    set(hwemu_link_command      "${hwemu_link_command}"     PARENT_SCOPE)
    set(swemu_link_command      "${swemu_link_command}"     PARENT_SCOPE)
    #---------------------------------------------------------------------------------------------------------
endfunction()


#Setting frequency override option.
if(KERNEL_CLOCK GREATER 0)
    if((${SDAccel_MAJOR_VERSION} EQUAL 2017) AND (${SDAccel_MINOR_VERSION} EQUAL 4))
        floatexpr("1000.0 / ${KERNEL_CLOCK}" KERNEL_PERIOD)
        message(WARNING "Bug workaround for SDx2017.4 : https://forums.xilinx.com/t5/SDAccel/How-to-set-the-target-frequency-for-the-kernel-in-SDACCL-2018-2/m-p/916088 https://forums.xilinx.com/t5/SDAccel/sdx2017-4-XOCC-kernel-compiler-cannot-set-kernel-frequency/td-p/894185")
        message("Calculated period for all of the kernels: ${KERNEL_PERIOD} nanoseconds")
        configure_file(scripts/KernelFreqSDx20174.tcl.in KernelFreqSDx20174.tcl)
        set(hw_link_command "${hw_link_command} --xp prop:solution.hls_pre_tcl=${CMAKE_BINARY_DIR}/KernelFreqSDx20174.tcl")
    else()
        message("Overriding default kernel clock frequency normally with the kernel_frequency option...")
        set(hw_link_command "${hw_link_command} --kernel_frequency ${KERNEL_CLOCK}")
    endif()
    set(KERNEL_CLOCK_RESOLVED ${KERNEL_CLOCK})
    set(KernelFrequency ${KERNEL_CLOCK})
else()
    set(KERNEL_CLOCK_RESOLVED 200)
    set(KernelFrequency ${KERNEL_CLOCK})
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/config)

file(WRITE "${CMAKE_BINARY_DIR}/Compile_Hw_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")
file(WRITE "${CMAKE_BINARY_DIR}/Compile_HwEmu_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")
file(WRITE "${CMAKE_BINARY_DIR}/Compile_SwEmu_Batch.sh" "find . -name \"*.xo\" -type f -delete\n")





#==========================================================================================================================================

set(SP_TAG_CONV2D
        "--sp task_conv2_1x1_direct_1.a:bank${CFG1_Conv2_DDRBANK_inputTn}\
 --sp task_conv2_1x1_direct_1.b:bank${CFG1_Conv2_DDRBANK_weightTn}\
 --sp task_conv2_1x1_direct_1.e:bank${CFG1_Conv2_DDRBANK_biasTn}\
 --sp task_conv2_1x1_direct_1.c:bank${CFG1_Conv2_DDRBANK_outputTn}")

set(SP_TAG_TOPK
        "--sp task_topk_1.inputTn:bank${CFG2_TopK_DDRBANK_inputTn}\
 --sp task_topk_1.indicesSplitedTn:bank${CFG2_TopK_DDRBANK_indicesSplitedTn}")

set(SP_TAG_MATOPS
        "--sp task_matops_1.inputTn1:bank${CFG3_MatOps_DDRBANK_inputTn1}\
 --sp task_matops_1.inputTn2:bank${CFG3_MatOps_DDRBANK_inputTn2}\
 --sp task_matops_1.outputTn:bank${CFG3_MatOps_DDRBANK_outputTn}")

set(SP_TAG_REDUCE
        "--sp task_reduce_1.inputTn:bank${CFG4_Reduce_DDRBANK_inputTn}\
 --sp task_reduce_1.outputTn:bank${CFG4_Reduce_DDRBANK_outputTn}")

set(SP_TAG_MATMUL
        "--sp task_matmul_1.inputTn1:bank${CFG5_MatMul_DDRBANK_inputTn1}\
 --sp task_matmul_1.inputTn2:bank${CFG5_MatMul_DDRBANK_inputTn2}\
 --sp task_matmul_1.outputTn:bank${CFG5_MatMul_DDRBANK_outputTn}")

set(SP_TAG_TILE
        "--sp task_tile_1.inputTn:bank${CFG6_Tile_DDRBANK_inputTn}\
 --sp task_tile_1.outputTn:bank${CFG6_Tile_DDRBANK_outputTn}")

set(SP_TAG_GATHER
        "--sp task_gather_1.inputTn:bank${CFG7_Gather_DDRBANK_inputTn}\
 --sp task_gather_1.indicesTn:bank${CFG7_Gather_DDRBANK_indicesTn}\
 --sp task_gather_1.outputTn:bank${CFG7_Gather_DDRBANK_outputTn}")

set(SP_TAG_CONCAT
        "--sp task_concat_1.inputTn1:bank${CFG8_Concat_DDRBANK_inputTn1}\
 --sp task_concat_1.inputTn2:bank${CFG8_Concat_DDRBANK_inputTn2}\
 --sp task_concat_1.outputTn:bank${CFG8_Concat_DDRBANK_outputTn}")

set(SP_TAG_TRANSPOSE
        "--sp task_transpose_1.inputTn:bank${CFG9_Transpose_DDRBANK_inputTn}\
 --sp task_transpose_1.outputTn:bank${CFG9_Transpose_DDRBANK_outputTn}")

set(SP_TAG_RELUSQRTSQUARE
        "--sp task_relu_sqrt_square_1.inputTn:bank${CFG10_ReluSqrtSquare_DDRBANK_inputTn}\
 --sp task_relu_sqrt_square_1.outputTn:bank${CFG10_ReluSqrtSquare_DDRBANK_outputTn}")

set(SP_TAG_DATAMOVER "")
if(${UseMemoryBank0})
    list(APPEND SP_TAG_DATAMOVER "--sp task_datamover_1.dataBank0:bank0")
endif()
if(${UseMemoryBank1})
    list(APPEND SP_TAG_DATAMOVER "--sp task_datamover_1.dataBank1:bank1")
endif()
if(${UseMemoryBank2})
    list(APPEND SP_TAG_DATAMOVER "--sp task_datamover_1.dataBank2:bank2")
endif()
if(${UseMemoryBank3})
    list(APPEND SP_TAG_DATAMOVER "--sp task_datamover_1.dataBank3:bank3")
endif()
string(REPLACE ";" " " SP_TAG_DATAMOVER "${SP_TAG_DATAMOVER}" )
message("DataMover SP Tag: ${SP_TAG_DATAMOVER}")

set(SP_TAG_PADUNPAD
        "--sp task_pad_unpad_1.inputTn:bank${CFG11_PadUnpad_DDRBANK_inputTn}\
 --sp task_pad_unpad_1.outputTn:bank${CFG11_PadUnpad_DDRBANK_outputTn}")



sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/conv2_1x1_direct.cpp"
        "conv2_1x1_direct"
        "task_conv2_1x1_direct"
        FALSE
        TRUE
        ${SP_TAG_CONV2D}
        "")
#sdaccel_target(
#   "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/topk_selectionsort.cpp"
#   "topk"
#   "task_topk"
#   FALSE
#   TRUE
#   ${SP_TAG_TOPK}
#   "")
#sdaccel_target(
#   "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/topk_insertionsort.cpp"
#   "topk"
#   "task_topk"
#   FALSE
#   TRUE
#   ${SP_TAG_TOPK}
#   "")
#sdaccel_target(
#   "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/topk_mergesort.cpp"
#   "topk"
#   "task_topk"
#   FALSE
#   TRUE
#   ${SP_TAG_TOPK}
#   "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/topk_mergesortdf_pe.cpp"
        "topk"
        "task_topk"
        FALSE
        TRUE
        ${SP_TAG_TOPK}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/matops.cpp"
        "matops"
        "task_matops"
        FALSE
        TRUE
        ${SP_TAG_MATOPS}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/reduce.cpp"
        "reduce"
        "task_reduce"
        FALSE
        TRUE
        ${SP_TAG_REDUCE}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/matmul.cpp"
        "matmul"
        "task_matmul"
        FALSE
        TRUE
        ${SP_TAG_MATMUL}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/tile.cpp"
        "tile"
        "task_tile"
        FALSE
        TRUE
        ${SP_TAG_TILE}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/gather.cpp"
        "gather"
        "task_gather"
        FALSE
        TRUE
        ${SP_TAG_GATHER}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/concat.cpp"
        "concat"
        "task_concat"
        FALSE
        TRUE
        ${SP_TAG_CONCAT}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/transpose.cpp"
        "transpose"
        "task_transpose"
        FALSE
        TRUE
        ${SP_TAG_TRANSPOSE}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/relu_sqrt_square.cpp"
        "relu_sqrt_square"
        "task_relu_sqrt_square"
        FALSE
        TRUE
        ${SP_TAG_RELUSQRTSQUARE}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/datamover.cpp"
        "datamover"
        "task_datamover"
        FALSE
        TRUE
        ${SP_TAG_DATAMOVER}
        "")
sdaccel_target(
        "${CMAKE_SOURCE_DIR}/src/fpga/xilinx/kernels/pad_unpad.cpp"
        "padunpad"
        "task_pad_unpad"
        FALSE
        TRUE
        ${SP_TAG_PADUNPAD}
        "")



# Generating bash script to compile the kernels to get *.xo files on a cmake custom target
add_custom_target(compile_hw COMMAND sh Compile_Hw_Batch.sh)
add_custom_target(compile_hwemu COMMAND sh Compile_HwEmu_Batch.sh)
add_custom_target(compile_swemu COMMAND sh Compile_SwEmu_Batch.sh)

set(SlrUtilOutputFile ${CMAKE_BINARY_DIR}/SLR_UTILIZATION.log)
set(PblockUtilOutputFile ${CMAKE_BINARY_DIR}/PBLOCK_UTILIZATION.log)
set(PreRouteCheckpointFile ${CMAKE_BINARY_DIR}/Pre_route_checkpoint.dcp)
set(PostRouteCheckpointFile ${CMAKE_BINARY_DIR}/Post_route_checkpoint.dcp)
set(PowerReportTextFile ${CMAKE_BINARY_DIR}/PowerReportPostRoute.log)
set(PowerReportRpxFile ${CMAKE_BINARY_DIR}/PowerReportPostRoute.rpx)
configure_file(${CMAKE_SOURCE_DIR}/scripts/PreRoute.tcl ${CMAKE_BINARY_DIR}/PreRoute.tcl)
configure_file(${CMAKE_SOURCE_DIR}/scripts/PostRoute.tcl ${CMAKE_BINARY_DIR}/PostRoute.tcl)
set(PreRouteFilePath ${CMAKE_BINARY_DIR}/PreRoute.tcl)
set(PostRouteFilePath ${CMAKE_BINARY_DIR}/PostRoute.tcl)

# Time to link *.xo files that have been generated from the kernels, to generate *.xclbin
if(ENABLE_PROFILING)
    if(PROFILE_DATA)
        set(swemu_link_command "${swemu_link_command} --profile_kernel data:all:all:all")
        set(hwemu_link_command "${hwemu_link_command} --profile_kernel data:all:all:all")
        set(hw_link_command "${hw_link_command} --profile_kernel data:all:all:all")
    endif()
    if(PROFILE_STALL)
        set(swemu_link_command "${swemu_link_command} --profile_kernel stall:all:all")
        set(hwemu_link_command "${hwemu_link_command} --profile_kernel stall:all:all")
        set(hw_link_command "${hw_link_command} --profile_kernel stall:all:all")
    endif()
    if(PROFILE_EXEC)
        set(swemu_link_command "${swemu_link_command} --profile_kernel exec:all:all")
        set(hwemu_link_command "${hwemu_link_command} --profile_kernel exec:all:all")
        set(hw_link_command "${hw_link_command} --profile_kernel data:all:all:all --profile_kernel stall:all:all --profile_kernel exec:all:all")
    endif()
endif()

file(WRITE "${CMAKE_BINARY_DIR}/Link_Hw_Batch.sh"
        "${SDAccel_XOCC} -g -l -t hw --platform ${DSA_NAME} --xp vivado_prop:run.impl_1.{STEPS.ROUTE_DESIGN.TCL.PRE}={${PreRouteFilePath}} --xp vivado_prop:run.impl_1.{STEPS.ROUTE_DESIGN.TCL.POST}={${PostRouteFilePath}} ${VivadoOptions} --report_level ${ReportLevel} --jobs ${CpuCount} ${hw_link_command} ${hw_link} -o ${CMAKE_PROJECT_NAME}_hw.xclbin"
        )

file(WRITE "${CMAKE_BINARY_DIR}/Link_HwEmu_Batch.sh"
        "${SDAccel_XOCC} -g -l -t hw_emu --platform ${DSA_NAME} --jobs ${CpuCount} ${hwemu_link_command} ${hwemu_link} -o ${CMAKE_PROJECT_NAME}_hw_emu.xclbin"
        )

file(WRITE "${CMAKE_BINARY_DIR}/Link_SwEmu_Batch.sh"
        "${SDAccel_XOCC} -g -l -t sw_emu --platform ${DSA_NAME} --jobs ${CpuCount} ${swemu_link_command} ${swemu_link} -o ${CMAKE_PROJECT_NAME}_sw_emu.xclbin"
        )
add_custom_target(link_hw COMMAND sh Link_Hw_Batch.sh)
add_custom_target(link_hwemu COMMAND sh Link_HwEmu_Batch.sh )
add_custom_target(link_swemu COMMAND sh Link_SwEmu_Batch.sh )


file(WRITE "${CMAKE_BINARY_DIR}/Interactrive_Hw_Batch.sh"
        "${SDAccel_XOCC} -g -l -t hw --platform ${DSA_NAME} ${hw_link_command} ${hw_link} -o ${CMAKE_PROJECT_NAME}_hw.xclbin --interactive synth"
        )
add_custom_target(interactive_hw COMMAND sh Interactrive_Hw_Batch.sh)

set(xcl_path_swemu "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_sw_emu.xclbin")
set(xcl_path_hwemu "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_hw_emu.xclbin")
set(xcl_path_hw "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}_hw.awsxclbin")
set(HostExecutablePath "${CMAKE_BINARY_DIR}/${HostExecutableName}")
set(DataDirectory "${CMAKE_SOURCE_DIR}/data")
configure_file(${CMAKE_SOURCE_DIR}/scripts/LaunchDeepPointV2FPGA.sh.in LaunchDeepPointV2FPGA.sh @ONLY)

configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_swemu.sh ${CMAKE_BINARY_DIR}/autobuild_swemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_hwemu.sh ${CMAKE_BINARY_DIR}/autobuild_hwemu.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/autobuild_hw.sh ${CMAKE_BINARY_DIR}/autobuild_hw.sh @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/PasteBinScript.py ${CMAKE_BINARY_DIR}/PasteBinScript.py @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/submodules/SimplePasteBin/SimplePasteBin.py ${CMAKE_BINARY_DIR}/SimplePasteBin.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/sdaccel.ini ${CMAKE_BINARY_DIR}/sdaccel.ini COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/scripts/gather_results.sh ${CMAKE_BINARY_DIR}/gather_results.sh COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/CMakeLists.txt ${CMAKE_BINARY_DIR}/CMakeLists_main.log COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/config/CMakeLists.txt ${CMAKE_BINARY_DIR}/CMakeLists_config.log COPYONLY)



include_directories(${CMAKE_SOURCE_DIR}/submodules/rapidjson/include)
add_executable(${HostExecutableName}
        src/main.cpp
        src/CTensorBase.cpp
        src/fpga/xilinx/CKernelWrapper.cpp
        src/fpga/xilinx/CKernelArg.cpp
        src/CProfiler.cpp
        src/fpga/xilinx/CImplementationXilinx.cpp
        src/CClassifierMultiPlatform.cpp
        src/models/CModel1.cpp
        src/CWeightLoader.cpp
        src/CPlatformSelection.cpp
        src/cpu/CImplementationCpu.cpp
        src/CImplementationBase.cpp
        src/cnpy.cpp
        src/fpga/xilinx/xcl2.cpp
        )

target_link_libraries(${HostExecutableName} ${SDAccel_LIBRARIES} ${SDAccel_FLOATING_POINT_LIBRARY} z stdc++fs spdlog)

add_subdirectory(${CMAKE_SOURCE_DIR}/submodules/googletest)
add_subdirectory(${CMAKE_SOURCE_DIR}/submodules/spdlog)
#add_subdirectory(${CMAKE_SOURCE_DIR}/test)


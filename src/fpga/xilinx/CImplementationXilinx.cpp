#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "fpga/xilinx/CImplementationXilinx.h"
#include <iostream>
#include <memory>
#include "GlobalHelpers.h"

using namespace std;

CImplementationXilinx::CImplementationXilinx(
    CProfiler *profiler,
    bool enableOclProfiling,
    bool logMemBankCrossings){

  m_iStatus = 0;
  m_ePlatform = PLATFORMS::XIL;
  m_ptrProfiler = profiler;
  m_bEnableOclProfiling = enableOclProfiling;
  m_bLogMemBankCrossings = logMemBankCrossings;
  ResetLayerIdCounter(0);

  //======================================================================================================================
  {
    const RUN_MODE mode = GetModeEnvVar();
    if(mode==RUN_MODE::Unknown){
      SPDLOG_LOGGER_WARN(logger,"XCL_EMULATION_MODE is not set. System run(real FPGA) is considered.");
      //assert(SetModeEnvVar(RUN_MODE::SwEmu)==0);
    }else{
      SPDLOG_LOGGER_TRACE(logger,"Mode: {}",
                          (mode==RUN_MODE::SwEmu?"Sw-emulation":
                           mode==RUN_MODE::HwEmu?"Hw-emulation":
                           "Hardware(FPGA)"));
    }
  }

  //======================================================================================================================
  {
    auto devices = xcl::get_xil_devices();
    SPDLOG_LOGGER_TRACE(logger,"Xilinx Devices Found: {}", devices.size());
    assert(devices.size()>0);

    SPDLOG_LOGGER_TRACE(logger,"Using device index 0");
    m_oDevice = devices[0];

    OclCheck(
        m_iStatus,
        m_ptrContext = new cl::Context(m_oDevice, NULL, NULL, NULL, &m_iStatus)
    );
    OclCheck(
        m_iStatus,
        m_ptrQueue = new cl::CommandQueue(
            *m_ptrContext,
            m_oDevice,
            (m_bEnableOclProfiling ? (CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) :
             CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
            &m_iStatus)
    );
    m_strDeviceName = m_oDevice.getInfo<CL_DEVICE_NAME>();
    SPDLOG_LOGGER_TRACE(logger,"Found Device: {}", m_strDeviceName.c_str());

    auto fileBuf = xcl::read_binary_file(globalArgXclBin);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    OclCheck(
        m_iStatus,
        m_ptrProgram = new cl::Program(
            *m_ptrContext,
            {m_oDevice},
            bins,
            NULL,
            &m_iStatus)
    );

    m_ptrXilInfo = new CXilinxInfo(m_ptrProgram, m_ptrContext, m_ptrQueue, m_bEnableOclProfiling);
    m_ptrDataMoverProfiledDataVec = new vector<ProfiledLaunchData>();
    m_ptrXilInfo->SetAccumulatedProfiledKernelLaunchDataVecPtr(m_ptrDataMoverProfiledDataVec);
#ifdef USEMEMORYBANK0
    m_ptrDataMoverDummyTensorBank0 = new CTensorXil<float>(m_ptrXilInfo, {5,1,1024}, true, 0);
#else
    m_ptrDataMoverDummyTensorBank0 = nullptr;
#endif
#ifdef USEMEMORYBANK1
    m_ptrDataMoverDummyTensorBank1 = new CTensorXil<float>(m_ptrXilInfo, {5,1,1024}, true, 1);
#else
    m_ptrDataMoverDummyTensorBank1 = nullptr;
#endif
#ifdef USEMEMORYBANK2
    m_ptrDataMoverDummyTensorBank2 = new CTensorXil<float>(m_ptrXilInfo, {50,1024,1024}, true, 2);
#else
    m_ptrDataMoverDummyTensorBank2 = nullptr;
#endif
#ifdef USEMEMORYBANK3
    m_ptrDataMoverDummyTensorBank3 = new CTensorXil<float>(m_ptrXilInfo, {50,1024,1024}, true, 3);
#else
    m_ptrDataMoverDummyTensorBank3 = nullptr;
#endif
    m_ptrXilInfo->SetDataMoverDummyTensors(
        m_ptrDataMoverDummyTensorBank0,
        m_ptrDataMoverDummyTensorBank1,
        m_ptrDataMoverDummyTensorBank2,
        m_ptrDataMoverDummyTensorBank3);
  }

  //======================================================================================================================
  m_ptrKernelConcat = std::make_unique<CKernelWrapperConcat>(
      "task_concat", "concat.cpp", m_ptrXilInfo,
      ConfigTaskConcat::BankIndex_inputTn1,
      ConfigTaskConcat::BankIndex_inputTn2,
      ConfigTaskConcat::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelMatmul = std::make_unique<CKernelWrapperMatmul>(
      "task_matmul", "matmul.cpp", m_ptrXilInfo,
      ConfigTaskMatMul::BankIndex_inputTn1,
      ConfigTaskMatMul::BankIndex_inputTn2,
      ConfigTaskMatMul::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelRss = std::make_unique<CKernelWrapperReluSqrtSquare>(
      "task_relu_sqrt_square", "relu_sqrt_square.cpp", m_ptrXilInfo,
      ConfigTaskReluSqrtSquare::BankIndex_inputTn,
      ConfigTaskReluSqrtSquare::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelBasicOps = std::make_unique<CKernelWrapperBasicOps>(
      "task_basicops", "basicops.cpp", m_ptrXilInfo,
      ConfigTaskBasicOps::BankIndex_inputTn1,
      ConfigTaskBasicOps::BankIndex_inputTn2,
      ConfigTaskBasicOps::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelTile = std::make_unique<CKernelWrapperTile>(
      "task_tile", "tile.cpp", m_ptrXilInfo,
      ConfigTaskTile::BankIndex_inputTn,
      ConfigTaskTile::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelTranspose = std::make_unique<CKernelWrapperTranspose>(
      "task_transpose", "transpose.cpp", m_ptrXilInfo,
      ConfigTaskTranspose::BankIndex_inputTn,
      ConfigTaskTranspose::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelGather = std::make_unique<CKernelWrapperGather>(
      "task_gather", "gather.cpp", m_ptrXilInfo,
      ConfigTaskGather::BankIndex_inputTn,
      ConfigTaskGather::BankIndex_indicesTn,
      ConfigTaskGather::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelReduce = std::make_unique<CKernelWrapperReduce>(
      "task_reduce", "reduce.cpp", m_ptrXilInfo,
      ConfigTaskReduce::BankIndex_inputTn,
      ConfigTaskReduce::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelPadUnpad = std::make_unique<CKernelWrapperPadUnpad>(
      "task_pad_unpad", "pad_unpad.cpp", m_ptrXilInfo,
      ConfigTaskPadUnpad::BankIndex_inputTn,
      ConfigTaskPadUnpad::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelTopK = std::make_unique<CKernelWrapperTopK>(
      "task_topk", "topk_mergesoftdf_pe.cpp", m_ptrXilInfo,
      ConfigTaskTopK::BankIndex_inputTn,
      ConfigTaskTopK::BankIndex_indicesSplitedTn,
      ConfigTaskTopK::MaxSliceLen,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);
  m_ptrKernelConv = std::make_unique<CKernelWrapperConv>(
      "task_conv2_1x1_direct", "conv2_1x1_direct.cpp", m_ptrXilInfo,
      ConfigTaskConv2::BankIndex_inputTn,
      ConfigTaskConv2::BankIndex_weightTn,
      ConfigTaskConv2::BankIndex_biasTn,
      ConfigTaskConv2::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bEnableOclProfiling,
      m_bLogMemBankCrossings);


}

int CImplementationXilinx::SetModeEnvVar(RUN_MODE &mode) {
  int result = 0;
  if(mode==RUN_MODE::Unknown) return -2;
  const char* strMode = mode==RUN_MODE::SwEmu? "sw_emu":
                        mode==RUN_MODE::HwEmu? "hw_emu":
                        "system";
  result = setenv("XCL_EMULATION_MODE", strMode, 1); // Env var override is enabled.

  if(result<0){
    cerr<<"SetModeEnvVar: Error setting XCL_EMULATION_MODE env. var."<<endl;
  }
  return result;
}

RUN_MODE CImplementationXilinx::GetModeEnvVar() const {
  if(const char *_xcl_mode = getenv("XCL_EMULATION_MODE")){
    const string xcl_mode = string(_xcl_mode);
    RUN_MODE mode =  xcl_mode=="sw_emu" ? RUN_MODE::SwEmu:
                     xcl_mode=="hw_emu" ? RUN_MODE::HwEmu:
                     xcl_mode=="system" ? RUN_MODE::Hw:
                     RUN_MODE::Unknown ;
    return mode;
  }else{
    return RUN_MODE::Unknown;
  }
}

const string CImplementationXilinx::GetOclErrorMessage(cl_int error) const
{
  switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

      // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

      // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
  }
}
CXilinxInfo *CImplementationXilinx::GetXilInfo() {
  return m_ptrXilInfo;
}
CImplementationXilinx::~CImplementationXilinx() {
  SPDLOG_LOGGER_TRACE(logger, "Waiting for async-queue to finish up in ~CImplementationXilinx().");
  OclCheck(m_iStatus, m_iStatus = m_ptrQueue->finish());

  SPDLOG_LOGGER_TRACE(logger, "Processing accumulated profiled kernel launches in ~CImplementationXilinx().");

  std::vector<std::vector<ProfiledLaunchData>> accumulatedProfiledKernelsData = {
      *m_ptrDataMoverProfiledDataVec,
      m_ptrKernelConcat->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelMatmul->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelRss->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelBasicOps->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelTile->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelTranspose->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelGather->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelReduce->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelPadUnpad->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelTopK->GetAccumulatedProfiledKernelLaunchData(),
      m_ptrKernelConv->GetAccumulatedProfiledKernelLaunchData()
  };

  for(auto &vecData:accumulatedProfiledKernelsData){
    if(vecData.size()!=0) {
      for (auto &data:vecData) {
        if(data.parentLayerId == DATAMOVER_ID){
          m_ptrProfiler->StartKernelDatamover(PLATFORMS::XIL, data.parentLayerId, data.optionalValue, data.taskName, data.durationOcl);
          m_ptrProfiler->FinishKernel();
        }else{
          m_ptrProfiler->StartKernel(PLATFORMS::XIL, data.parentLayerId, data.taskName, data.durationOcl);
          m_ptrProfiler->FinishKernel();
        }
      }
    }
  }

  SPDLOG_LOGGER_TRACE(logger, "Destroying CImplementationXilinx().");

  delete m_ptrXilInfo;
  delete m_ptrDataMoverProfiledDataVec;
  delete m_ptrProgram;
  delete m_ptrContext;
  delete m_ptrQueue;

  SPDLOG_LOGGER_TRACE(logger, "Destroyed CImplementationXilinx().");
}
CTensorBasePtr CImplementationXilinx::Concat2(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, unsigned concatAxis) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape1",inputTn1->GetShape()},{"shape2",inputTn2->GetShape()}}),
      new CProfiler::DictIntPtr({{"concatAxis",concatAxis}}),
      nullptr);

  ValidateTensorPlatforms({inputTn1,inputTn2}, PLATFORMS::XIL);

  CTensorBasePtr outputTn =
      m_ptrKernelConcat->EnqueueKernelLaunch(GetTheLastLayerId(), inputTn1, inputTn2, concatAxis);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::MatMul(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape1",inputTn1->GetShape()},{"shape2",inputTn2->GetShape()}}),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn1,inputTn2}, PLATFORMS::XIL);

  CTensorBasePtr outputTn =
      m_ptrKernelMatmul->EnqueueKernelLaunch(GetTheLastLayerId(), inputTn1, inputTn2);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::ReLU(CTensorBasePtr inputTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn =
      m_ptrKernelRss->EnqueueKernelLaunch(
          GetTheLastLayerId(), inputTn, true, false, false);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Sqrt(CTensorBasePtr inputTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn =
      m_ptrKernelRss->EnqueueKernelLaunch(
          GetTheLastLayerId(), inputTn, false, true, false);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Square(CTensorBasePtr inputTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelRss->EnqueueKernelLaunch(
      GetTheLastLayerId(), inputTn, false, false, true);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::BasicOps(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, BASIC_OPS mode) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape1",inputTn1->GetShape()},{"shape2",inputTn2->GetShape()}}),
      new CProfiler::DictIntPtr({{
                                     "mode",
                                     mode==BASIC_OPS::ADD ? 0 : mode==BASIC_OPS::SUB ? 1 : mode==BASIC_OPS::MUL_ELEMENTWISE ? 2 : 3}}),
      nullptr);

  ValidateTensorPlatforms({inputTn1,inputTn2}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelBasicOps->EnqueueKernelLaunch(
      GetTheLastLayerId(),
      inputTn1,
      inputTn2,
      mode,
      false,
      0.0f);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::BasicOps(CTensorBasePtr inputTn1, float scalar, BASIC_OPS mode) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape1",inputTn1->GetShape()},{"shape2",{1}}}),
      new CProfiler::DictIntPtr({{
                                     "mode",
                                     mode==BASIC_OPS::ADD ? 0 :
                                       mode==BASIC_OPS::SUB ? 1 :
                                       mode==BASIC_OPS::MUL_ELEMENTWISE ? 2 : 3}}),
      new CProfiler::DictFloatPtr({{"scalar",scalar}}));

  ValidateTensorPlatforms({inputTn1}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelBasicOps->EnqueueKernelLaunch(
      GetTheLastLayerId(),
      inputTn1,
      nullptr,
      mode,
      true,
      scalar);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Tile(CTensorBasePtr inputTn, unsigned tileAxis, unsigned tileCount) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({{"tileAxis",tileAxis},{"tileCount",tileCount}}),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelTile->EnqueueKernelLaunch(
      GetTheLastLayerId(), inputTn, tileAxis, tileCount);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Transpose(CTensorBasePtr inputTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelTranspose->EnqueueKernelLaunch(GetTheLastLayerId(), inputTn);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Gather(CTensorBasePtr inputTn, CTensorBasePtr indicesTn, unsigned indicesOfAxis) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}, {"shape.indices",indicesTn->GetShape()}}),
      new CProfiler::DictIntPtr({{"indicesOfAxis",indicesOfAxis}}),
      nullptr);

  ValidateTensorPlatforms({inputTn,indicesTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelGather->EnqueueKernelLaunch(GetTheLastLayerId(), inputTn, indicesTn, indicesOfAxis);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Reduce(CTensorBasePtr inputTn,
                                             REDUCTION_OPS mode,
                                             unsigned powY,
                                             const std::vector<unsigned> &combination) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({
        {"shape",inputTn->GetShape()},
        {"combination",combination},
        }),
      new CProfiler::DictIntPtr({
        {"reduction_op",
         mode==REDUCTION_OPS::SUM?0:
         mode==REDUCTION_OPS::MAX?1:
         -1
        },
        {"powY",powY},
        {"rank",inputTn->GetRank()},
      }),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelReduce->EnqueueKernelLaunch(
      GetTheLastLayerId(),
      inputTn,
      mode,
      powY,
      combination
  );

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Mean(CTensorBasePtr inputTn,
                                           const std::vector<unsigned> &combination) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({
        {"shape",inputTn->GetShape()},
        {"combination",combination},
      }),
      new CProfiler::DictIntPtr({
                                    {"rank",inputTn->GetRank()}
      }),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);
  ConditionCheck(inputTn->GetRank()==2 || inputTn->GetRank()==4, "Only tensors of ranks 2 and 4 are supported.");
  ConditionCheck(inputTn->GetRank()==combination.size(), "The combination's size must be equal to the input tensor's rank.");

  if(inputTn->GetRank()==4){
    ConditionCheck(
        (combination[0] && combination[1] && combination[2] && !combination[3]),
        "Unsupported combination for the input tensor."
    );
  }
  if(inputTn->GetRank()==2){
    ConditionCheck(
        (combination[0] && !combination[1]),
        "Unsupported combination for the input tensor."
    );
  }

  auto localCombination = combination;
  unsigned diff = inputTn->ExpandDimZeroToRank(4);
  for(unsigned d=0; d<diff; d++){
    localCombination.insert(localCombination.begin(),1); // we dont insert 0 here as we want to use the TTTF reduction kernel.
  }

  CTensorBasePtr reducedTn = Reduce(inputTn, REDUCTION_OPS::SUM, 1, localCombination);
  float coef = (float)inputTn->GetLen() / (float)reducedTn->GetLen(); // dim0xdim1xdim2 (for TTTF)
  CTensorBasePtr outputTn = BasicOps(reducedTn, coef, BASIC_OPS::DIV_ELEMENTWISE);

  inputTn->SqueezeDimZeroTimesTry(diff);
  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Variance(CTensorBasePtr inputTn,
                                               const std::vector<unsigned> &combination) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({
                                  {"shape",inputTn->GetShape()},
                                  {"combination",combination}
                                }),
      new CProfiler::DictIntPtr({
                                    {"rank",inputTn->GetRank()}
                                }),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);
  ConditionCheck(inputTn->GetRank()==2 || inputTn->GetRank()==4, "Only tensors of ranks 2 and 4 are supported.");
  ConditionCheck(combination.size()==inputTn->GetRank(), "The combination's size must be equal to the input tensor's rank.");
  if(inputTn->GetRank()==4){
    ConditionCheck(
        (combination[0] && combination[1] && combination[2] && !combination[3]),
        "Unsupported combination for the input tensor."
    );
  }
  if(inputTn->GetRank()==2){
    ConditionCheck(
        (combination[0] && !combination[1]),
        "Unsupported combination for the input tensor."
    );
  }
  auto localCombination = combination;

  unsigned diff = inputTn->ExpandDimZeroToRank(4);
  for(unsigned d=0; d<diff; d++){
    localCombination.insert(localCombination.begin(),1); // we do not insert 0 here as we want to use the TTTF reduction kernel.
  }

  CTensorBasePtr tmpTn = Reduce(inputTn, REDUCTION_OPS::SUM, 1, localCombination);
  CTensorBasePtr varianceXi2Tn = Reduce(inputTn, REDUCTION_OPS::SUM, 2, localCombination);
  float coef = (float)inputTn->GetLen() / (float)tmpTn->GetLen();
  CTensorBasePtr meanTn = BasicOps(tmpTn, coef, BASIC_OPS::DIV_ELEMENTWISE);
  CTensorBasePtr tmp2Tn = BasicOps(varianceXi2Tn, coef, BASIC_OPS::DIV_ELEMENTWISE);
  CTensorBasePtr tmp3Tn = BasicOps(meanTn, meanTn, BASIC_OPS::MUL_ELEMENTWISE);
  CTensorBasePtr outputTn = BasicOps(tmp2Tn, tmp3Tn, BASIC_OPS::SUB);

  inputTn->SqueezeDimZeroTimesTry(diff);
  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::PadLastDim(CTensorBasePtr inputTn, unsigned lastDimPadded) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({{"lastDimPadded",lastDimPadded}}),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelPadUnpad->EnqueueKernelLaunch(
      GetTheLastLayerId(), inputTn, true, false, lastDimPadded, 0);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::UnpadLastDim(CTensorBasePtr inputTn, unsigned lastDimUnpadded) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({{"lastDimUnpadded",lastDimUnpadded}}),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelPadUnpad->EnqueueKernelLaunch(
      GetTheLastLayerId(), inputTn, false, true, 0, lastDimUnpadded);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::TopK(CTensorBasePtr inputTn, unsigned axis, unsigned k) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({{"shape",inputTn->GetShape()}}),
      new CProfiler::DictIntPtr({
        {"axis",axis},
        {"k",k},
      }),
      nullptr);

  ValidateTensorPlatforms({inputTn}, PLATFORMS::XIL);

  CTensorBasePtr outputTn = m_ptrKernelTopK->EnqueueKernelLaunch(
      GetTheLastLayerId(), inputTn, axis, k);

  m_ptrProfiler->FinishLayer();
  return outputTn;
}
CTensorBasePtr CImplementationXilinx::Conv2D(CTensorBasePtr inputTn, CTensorBasePtr weightTn, CTensorBasePtr biasTn) {
  m_ptrProfiler->StartLayer(
      GetPlatform(),
      GenerateLayerId(),
      __func__,
      new CProfiler::DictShapePtr({
        {"shape.i",inputTn->GetShape()},
        {"shape.w",weightTn->GetShape()},
        {"shape.b",biasTn->GetShape()}
        }),
      nullptr,
      nullptr);

  ValidateTensorPlatforms({inputTn,weightTn,biasTn}, PLATFORMS::XIL);

  const auto shapeInput = inputTn->GetShape();
  const auto shapeWeight = weightTn->GetShape();

  const unsigned B  = shapeInput[0];
  const unsigned N  = shapeInput[1];
  const unsigned K  = shapeInput[2];
  const unsigned D1 = shapeInput[3];
  const unsigned D2 = shapeWeight[3];
  unsigned D1Padded=0;
  unsigned D2Padded=0;

  const unsigned int vecSizeTranspose = ConfigTaskConv2::kTransposeWidthBytes / CONFIG_DTYPE_SIZE;
  SPDLOG_LOGGER_DEBUG(logger, "vecSizeTranspose: {}", vecSizeTranspose);

  //-----------------------------------------------------------------
  // Padding inputTn
  // This block is disabled, as all the inputs are considered last dim padded already.(not in shape but in data layout)

  //-----------------------------------------------------------------
  // Padding weightTn
  CTensorBasePtr _weightPadded;
  if(D2%ConfigTaskConv2::kOuterTileSizeM!=0){
    //Super-vec Padding( 64->128 )
    D2Padded = DivCeil<unsigned>(D2, ConfigTaskConv2::kOuterTileSizeM)*ConfigTaskConv2::kOuterTileSizeM;
    _weightPadded = PadLastDim(weightTn, D2Padded);

    // The kernel is modified to not require the weight tensor to be
    // padded in dimension-zero.
    SPDLOG_LOGGER_DEBUG(logger, "Padding weightTn(super-vec padding):");
    SPDLOG_LOGGER_DEBUG(logger, "D2: {}", D2);
    SPDLOG_LOGGER_DEBUG(logger, "D2Padded: {}", D2Padded);

  }else{
    SPDLOG_LOGGER_DEBUG(logger, "Bypassing super-vec padding for weightTn");
    _weightPadded = weightTn;
    D2Padded = D2;
  }

  //-----------------------------------------------------------------
  auto outputPaddedTn =
  m_ptrKernelConv->EnqueueKernelLaunch(
      GetTheLastLayerId(),  /// TODO: This will cause two kernel launches (padunpad above and conv on this line) to have same id's.
      inputTn,
      _weightPadded,
      biasTn,
      B, N, K, D1, D2Padded
  );

  //-----------------------------------------------------------------
  // Unpadding outputPaddedTn
  CTensorBasePtr outputTn;
  if(D2%ConfigTaskConv2::kOuterTileSizeM!=0){
    outputTn = UnpadLastDim(outputPaddedTn, D2); // Super-vec Unpadding( 128->64 )
    SPDLOG_LOGGER_DEBUG(logger, "Unpadding the results (super-vec unpadding)");
  }else{
    outputTn = outputPaddedTn;
    SPDLOG_LOGGER_DEBUG(logger, "Bypassing super-vec unpadding of the results");
  }

  //-----------------------------------------------------------------
  m_ptrProfiler->FinishLayer();
  return outputTn;
}

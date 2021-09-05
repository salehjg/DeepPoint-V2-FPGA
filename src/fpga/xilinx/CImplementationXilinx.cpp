#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "fpga/xilinx/CImplementationXilinx.h"
#include <iostream>
#include "GlobalHelpers.h"

using namespace std;

CImplementationXilinx::CImplementationXilinx(bool profileOcl, CProfiler *profiler) {
  m_bOclProfileEnabled = profileOcl;
  m_ePlatform = PLATFORMS::XIL;
  m_ptrProfiler = profiler;
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
            (m_bOclProfileEnabled?(CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE):
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

    m_ptrXilInfo = new CXilinxInfo(m_ptrProgram,m_ptrContext,m_ptrQueue, m_bOclProfileEnabled);
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
  m_ptrKernelConcat = new CKernelWrapperConcat(
      "task_concat","concat.cpp",m_ptrXilInfo,
      ConfigTaskConcat::BankIndex_inputTn1,
      ConfigTaskConcat::BankIndex_inputTn2,
      ConfigTaskConcat::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bOclProfileEnabled);
  m_ptrKernelMatmul = new CKernelWrapperMatmul(
      "task_matmul","matmul.cpp",m_ptrXilInfo,
      ConfigTaskMatMul::BankIndex_inputTn1,
      ConfigTaskMatMul::BankIndex_inputTn2,
      ConfigTaskMatMul::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bOclProfileEnabled);
  m_ptrKernelRss = new CKernelWrapperReluSqrtSquare(
      "task_relu_sqrt_square","relu_sqrt_square.cpp",m_ptrXilInfo,
      ConfigTaskReluSqrtSquare::BankIndex_inputTn,
      ConfigTaskReluSqrtSquare::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bOclProfileEnabled);
  m_ptrKernelBasicOps = new CKernelWrapperBasicOps(
      "task_basicops","basicops.cpp",m_ptrXilInfo,
      ConfigTaskBasicOps::BankIndex_inputTn1,
      ConfigTaskBasicOps::BankIndex_inputTn2,
      ConfigTaskBasicOps::BankIndex_outputTn,
      KERNEL_DIR, KERNEL_ENABLED,
      m_bOclProfileEnabled);


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
      m_ptrKernelConcat->GetAccumulatedProfiledKernelLaunchData()
  };

  for(auto &vecData:accumulatedProfiledKernelsData){
    if(vecData.size()!=0) {
      for (auto &data:vecData) {
        m_ptrProfiler->StartKernel(PLATFORMS::XIL, data.parentLayerId, data.taskName, data.durationOcl);
        m_ptrProfiler->FinishKernel();
      }
    }
  }

  SPDLOG_LOGGER_TRACE(logger, "Destroying CImplementationXilinx().");

  delete m_ptrKernelConcat;
  delete m_ptrXilInfo;
  delete m_ptrDataMoverProfiledDataVec;
  delete m_ptrProgram;
  delete m_ptrContext;
  delete m_ptrQueue;

  SPDLOG_LOGGER_TRACE(logger, "Destroyed CImplementationXilinx().");
}
CTensorBasePtr CImplementationXilinx::Concat2(CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis) {
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

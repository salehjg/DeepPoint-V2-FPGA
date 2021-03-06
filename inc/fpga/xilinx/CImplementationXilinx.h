#pragma once

#include <string>
#include "CImplementationBase.h"
#include "fpga/xilinx/xcl2.h"
#include "CProfiler.h"
#include "CXilinxInfo.h"
#include "fpga/xilinx/kernels/CKernelWrapperConcat.h"
#include "fpga/xilinx/kernels/CKernelWrapperMatmul.h"
#include "fpga/xilinx/kernels/CKernelWrapperReluSqrtSquare.h"
#include "fpga/xilinx/kernels/CKernelWrapperBasicOps.h"
#include "fpga/xilinx/kernels/CKernelWrapperTile.h"
#include "fpga/xilinx/kernels/CKernelWrapperTranspose.h"
#include "fpga/xilinx/kernels/CKernelWrapperGather.h"
#include "fpga/xilinx/kernels/CKernelWrapperReduce.h"
#include "fpga/xilinx/kernels/CKernelWrapperPadUnpad.h"
#include "fpga/xilinx/kernels/CKernelWrapperTopK.h"
#include "fpga/xilinx/kernels/CKernelWrapperConv.h"

enum class RUN_MODE{
  SwEmu,
  HwEmu,
  Hw,
  Unknown
};

#define KERNEL_ENABLED true
#define KERNEL_DISABLED true

class CImplementationXilinx: public CImplementationBase {
 public:

  CImplementationXilinx(CProfiler *profiler, bool enableOclProfiling, bool logMemBankCrossings);
  ~CImplementationXilinx();
  CXilinxInfo* GetXilInfo();
  int SetModeEnvVar(RUN_MODE &mode);
  RUN_MODE GetModeEnvVar() const;
  const std::string GetOclErrorMessage(cl_int error) const;

  CTensorBasePtr Concat2      (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, unsigned concatAxis) override;
  CTensorBasePtr MatMul       (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) override;
  CTensorBasePtr ReLU         (CTensorBasePtr inputTn) override;
  CTensorBasePtr Sqrt         (CTensorBasePtr inputTn) override;
  CTensorBasePtr Square       (CTensorBasePtr inputTn) override;
  CTensorBasePtr BasicOps     (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, BASIC_OPS mode) override;
  CTensorBasePtr BasicOps     (CTensorBasePtr inputTn1, float scalar, BASIC_OPS mode) override;
  CTensorBasePtr Tile         (CTensorBasePtr inputTn, unsigned tileAxis, unsigned tileCount) override;
  CTensorBasePtr Transpose    (CTensorBasePtr inputTn) override;
  CTensorBasePtr Gather       (CTensorBasePtr inputTn, CTensorBasePtr indicesTn, unsigned indicesOfAxis) override;
  CTensorBasePtr Reduce       (CTensorBasePtr inputTn, REDUCTION_OPS mode, unsigned powY, const std::vector<unsigned> &combination) override;
  CTensorBasePtr Mean         (CTensorBasePtr inputTn, const std::vector<unsigned> &combination) override ;
  CTensorBasePtr Variance     (CTensorBasePtr inputTn, const std::vector<unsigned> &combination) override ;
  CTensorBasePtr PadLastDim   (CTensorBasePtr inputTn, unsigned lastDimPadded) override ;
  CTensorBasePtr UnpadLastDim (CTensorBasePtr inputTn, unsigned lastDimUnpadded) override ;
  CTensorBasePtr TopK         (CTensorBasePtr inputTn, unsigned axis, unsigned k) override ;
  CTensorBasePtr Conv2D       (CTensorBasePtr inputTn, CTensorBasePtr weightTn, CTensorBasePtr biasTn) override ;

 private:
  bool m_bEnableOclProfiling, m_bLogMemBankCrossings;
  CXilinxInfo *m_ptrXilInfo;
  cl_int m_iStatus;
  const std::string KERNEL_DIR = REPO_DIR "src/fpga/xilinx/kernels";
  std::string m_strDeviceName;
  cl::Device m_oDevice;
  cl::Context *m_ptrContext;
  cl::Program *m_ptrProgram;
  cl::CommandQueue *m_ptrQueue;
  CTensorXil<float> *m_ptrDataMoverDummyTensorBank0;
  CTensorXil<float> *m_ptrDataMoverDummyTensorBank1;
  CTensorXil<float> *m_ptrDataMoverDummyTensorBank2;
  CTensorXil<float> *m_ptrDataMoverDummyTensorBank3;
  std::vector<ProfiledLaunchData> *m_ptrDataMoverProfiledDataVec;

  std::unique_ptr<CKernelWrapperConcat>         m_ptrKernelConcat;
  std::unique_ptr<CKernelWrapperMatmul>         m_ptrKernelMatmul;
  std::unique_ptr<CKernelWrapperReluSqrtSquare> m_ptrKernelRss;
  std::unique_ptr<CKernelWrapperBasicOps>       m_ptrKernelBasicOps;
  std::unique_ptr<CKernelWrapperTile>           m_ptrKernelTile;
  std::unique_ptr<CKernelWrapperTranspose>      m_ptrKernelTranspose;
  std::unique_ptr<CKernelWrapperGather>         m_ptrKernelGather;
  std::unique_ptr<CKernelWrapperReduce>         m_ptrKernelReduce;
  std::unique_ptr<CKernelWrapperPadUnpad>       m_ptrKernelPadUnpad;
  std::unique_ptr<CKernelWrapperTopK>           m_ptrKernelTopK;
  std::unique_ptr<CKernelWrapperConv>           m_ptrKernelConv;
};



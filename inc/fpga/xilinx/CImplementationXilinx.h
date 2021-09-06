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

  CImplementationXilinx(bool profileOcl, CProfiler *profiler);
  ~CImplementationXilinx();
  CXilinxInfo* GetXilInfo();
  int SetModeEnvVar(RUN_MODE &mode);
  RUN_MODE GetModeEnvVar() const;
  const std::string GetOclErrorMessage(cl_int error) const;

  CTensorBasePtr Concat2      (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, int concatAxis) override;
  CTensorBasePtr MatMul       (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2) override;
  CTensorBasePtr ReLU         (CTensorBasePtr inputTn) override;
  CTensorBasePtr Sqrt         (CTensorBasePtr inputTn) override;
  CTensorBasePtr Square       (CTensorBasePtr inputTn) override;
  CTensorBasePtr BasicOps     (CTensorBasePtr inputTn1, CTensorBasePtr inputTn2, BASIC_OPS mode) override;
  CTensorBasePtr BasicOps     (CTensorBasePtr inputTn1, float scalar, BASIC_OPS mode) override;
  CTensorBasePtr Tile         (CTensorBasePtr inputTn, unsigned tileAxis, unsigned tileCount) override;
  CTensorBasePtr Transpose    (CTensorBasePtr inputTn) override;
  CTensorBasePtr Gather       (CTensorBasePtr inputTn, CTensorBasePtr indicesTn, unsigned indicesOfAxis) override;
  CTensorBasePtr Reduce       (CTensorBasePtr inputTn, REDUCTION_OPS mode, unsigned powY, bool overAxis0, bool overAxis1, bool overAxis2, bool overAxis3) override;

 private:
  bool m_bOclProfileEnabled;
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

  CKernelWrapperConcat *m_ptrKernelConcat;
  CKernelWrapperMatmul *m_ptrKernelMatmul;
  CKernelWrapperReluSqrtSquare *m_ptrKernelRss;
  CKernelWrapperBasicOps *m_ptrKernelBasicOps;
  CKernelWrapperTile *m_ptrKernelTile;
  CKernelWrapperTranspose *m_ptrKernelTranspose;
  CKernelWrapperGather *m_ptrKernelGather;
  CKernelWrapperReduce *m_ptrKernelReduce;
};



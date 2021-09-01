#pragma once

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE


#include "fpga/xilinx/CXilinxInfo.h"
#include "cpu/CTensor.h"
#include "CTensorBase.h"
#include "xilinx/config.h"
#include "fpga/xilinx/xcl2.h"
#include <memory>
#include "CStringFormatter.h"
#include <typeinfo>
#include <algorithm>
#include <string>
#include "GlobalHelpers.h"
#include "fpga/xilinx/AxiHelper.h"
#include <algorithm>


template <typename T>
class CTensorXil: public CTensorBase {
 public:
  using Ptr = std::shared_ptr<CTensorXil>;

  CTensorXil(const CTensorXil<T>& other);
  CTensorXil& operator=(const CTensorXil<T>& other);
  CTensorXil(CXilinxInfo *xilInfo, const std::vector<unsigned> &shape, bool fillZeros, int bank=-1, int axiWidth = CONFIG_M_AXI_WIDTH);
  CTensorXil(CXilinxInfo *xilInfo, const std::vector<unsigned> &shape, const T* hostBuff, int bank=-1, int axiWidth = CONFIG_M_AXI_WIDTH);
  CTensorXil(CXilinxInfo *xilInfo, const CTensor<T> &hostTn, int bank=-1, int axiWidth = CONFIG_M_AXI_WIDTH);
  CTensorXil<T>* CloneIfNeededToBank(const unsigned destBank);
  std::string GetTensorTag() const;
  CXilinxInfo *GetXilInfo() const;
  void SetTensorTag(std::string &tag);
  int GetDramBank() const;
  int GetAxiWidth() const;
  cl::Buffer& GetDeviceBuffer();
  unsigned GetLenPadded() const;
  unsigned long GetSizeBytes() const override ;
  unsigned GetSizeBytesPadded() const;
  unsigned GetVectorCountPadded() const;
  std::vector<unsigned> GetShapePadded() const;
  cl::Event* GetEventPtr();
  CTensor<T>* TransferToHost();

 private:
  static void EventCallback(cl_event event, cl_int execStatus, void *userData);
  void CloneFrom(const CTensorXil<T> &other);
  void CloneFrom(CXilinxInfo *xilInfo, const std::vector<unsigned> &shape, const T* unsafeHostBuff, int bank, int axiWidth, cl_bool isBlocking=CL_BLOCKING);
  void CloneFrom(CXilinxInfo *xilInfo, const std::vector<unsigned> &shape, int bank, int axiWidth);
  int TranslateBankIndex(int bankIndex);
  void ValidateBankIndex(int bankIndex);
  cl_mem_ext_ptr_t CreateExtendedPointer(void *hostPtr, cl_mem_flags memoryBank);
  T* PadHostBuffer(const std::vector<unsigned> &actualShape, const T *hostSrcBuff, int axiWidth);
  T* UnPadHostBuffer(const std::vector<unsigned> &actualShape, const T *hostSrcBuff, int axiWidth);
  std::vector<unsigned> PadShape(const std::vector<unsigned> &shape, int axiWidth) const;

#ifdef USEMEMORYBANK0
  int m_iDramBank = 0;
#else
  #ifdef USEMEMORYBANK1
        int m_iDramBank = 1;
    #else
        #ifdef USEMEMORYBANK2
            int m_iDramBank = 2;
        #else
            #ifdef USEMEMORYBANK3
                int m_iDramBank = 3;
            #else
                assert(false);
            #endif
        #endif
    #endif
#endif
  CXilinxInfo *m_ptrXilInfo;
  unsigned m_iAxiWidth;
  std::string m_strTensorTag = CStringFormatter()<<"Bank"<<m_iDramBank; // the default tn tag
  cl::Buffer m_oDeviceBuffer;
  cl_int m_iOclStatus;
  cl::Event m_oEvent;
  std::unique_ptr<T[]> m_ptrHostBuffForFillZero; // for async fill zero operation in the constructor
  std::unique_ptr<CallbackData[]> m_ptrCallBackData;
};











template <typename T>
void CTensorXil<T>::EventCallback(cl_event event, cl_int execStatus, void *userData) {
  if( execStatus != CL_COMPLETE ) {
    std::cout<<"ERROR IN KERNEL EXECUTION\n";
    ///TODO REPORT ERROR WITH SPDLOGGER
    return;
  }

  cl_ulong deviceTimeStart=0, deviceTimeEnd=0; //nano-seconds

  if(((CallbackData *) userData)->profileKernel){
    cl_int stat;

    OclCheck(stat, stat=clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(deviceTimeStart), &deviceTimeStart, nullptr));
    OclCheck(stat, stat=clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(deviceTimeEnd), &deviceTimeEnd, nullptr));

    cl_ulong durationNanoSeconds = deviceTimeEnd - deviceTimeStart;

    std::cout<<"KERNEL: ns:"<<durationNanoSeconds<<std::endl;

    auto *classPtr = static_cast<CXilinxInfo*>(((CallbackData *)userData)->classPtr);
    classPtr->AddProfiledDataMoverLaunchDetails("task_datamover", ((CallbackData *) userData)->parentLayerId, durationNanoSeconds);
  }
}

template <typename T>
void CTensorXil<T>::ValidateBankIndex(int bankIndex){
  if(bankIndex!=-1){
#ifndef USEMEMORYBANK0
    //assert(bankIndex!=0);
    if(bankIndex==0)
      throw std::runtime_error(CStringFormatter()<< __func__ << ": Xilinx DDR Bank0 is disabled and should not be used.");
#endif
#ifndef USEMEMORYBANK1
    //assert(bankIndex!=1);
    if(bankIndex==1)
      throw std::runtime_error(CStringFormatter()<< __func__ << ": Xilinx DDR Bank1 is disabled and should not be used.");
#endif
#ifndef USEMEMORYBANK2
    assert(bankIndex!=2);
    if(bankIndex==2)
      throw std::runtime_error(CStringFormatter()<< __func__ << ": Xilinx DDR Bank2 is disabled and should not be used.");
#endif
#ifndef USEMEMORYBANK3
    assert(bankIndex!=3);
    if(bankIndex==3)
      throw std::runtime_error(CStringFormatter()<< __func__ << ": Xilinx DDR Bank3 is disabled and should not be used.");
#endif
  }
}

template <typename T>
int CTensorXil<T>::TranslateBankIndex(int bankIndex){
  switch(bankIndex){
    case 0:{
      return XCL_MEM_DDR_BANK0;
    }break;
    case 1:{
      return XCL_MEM_DDR_BANK1;
    }break;
    case 2:{
      return XCL_MEM_DDR_BANK2;
    }break;
    case 3:{
      return XCL_MEM_DDR_BANK3;
    }break;
  };
}

template <typename T>
cl_mem_ext_ptr_t CTensorXil<T>::CreateExtendedPointer(void *hostPtr, cl_mem_flags memoryBank){
  cl_mem_ext_ptr_t extendedPointer;
  extendedPointer.flags = memoryBank;
  extendedPointer.obj = hostPtr;
  extendedPointer.param = 0;
  return extendedPointer;
}

template<typename T>
int CTensorXil<T>::GetAxiWidth() const {
  return m_iAxiWidth;
}

template<typename T>
cl::Buffer& CTensorXil<T>::GetDeviceBuffer() {
  return m_oDeviceBuffer;
}

/*!
 * Deep copies the content of the `other` xil tensor to the new instance.
 * @tparam T
 * @param other
 */
template<typename T>
CTensorXil<T>::CTensorXil(const CTensorXil<T> &other) {
  CloneFrom(other);
}

/*!
 * Deep copies the content of the `other` tensor to the new instance.
 * @tparam T
 * @param other
 * @return
 */
template<typename T>
CTensorXil<T>& CTensorXil<T>::operator=(const CTensorXil<T> &other) {
  // this = other !
  CloneFrom(other);
  return *this;
}

/*!
 * Creates a new instance with or without having the device memory initialized to zero.
 * Please note that the zero-filling is implemented non-blocking, enforcing the class to hold on to the host buffer for
 * zero-filling process (`m_ptrHostBuffForFillZero`).
 * @tparam T
 * @param context
 * @param queue
 * @param shape
 * @param fillZeros
 * @param bank
 * @param axiWidth
 */
template<typename T>
CTensorXil<T>::CTensorXil(CXilinxInfo *xilInfo,
                          const std::vector<unsigned> &shape,
                          bool fillZeros,
                          int bank,
                          int axiWidth) {
  SetPlatform(PLATFORMS::XIL);
  ValidateBankIndex(bank);
  m_iDramBank = bank==-1? m_iDramBank : bank;
  m_iAxiWidth = axiWidth;
  m_ptrXilInfo = xilInfo;
  SetShape(shape);
  if(fillZeros){
    m_ptrHostBuffForFillZero.reset(new T[GetLenPadded()]);
    for(unsigned i=0; i<GetLenPadded(); i++){ m_ptrHostBuffForFillZero[i]=0;}
    CloneFrom(xilInfo,shape,m_ptrHostBuffForFillZero.get(),bank,axiWidth,CL_NON_BLOCKING);
  }else{
    CloneFrom(xilInfo,shape,bank,axiWidth);
  }
}

/*!
 * Pads the hostBuff according to `axiWidth` in order for the device side buffer to
 * comply with the padded last dim policy.
 * host-device transfers are blocking.
 * @tparam T
 * @param context
 * @param queue
 * @param shape
 * @param hostBuff
 * @param bank
 * @param axiWidth
 */
template<typename T>
CTensorXil<T>::CTensorXil(CXilinxInfo *xilInfo,
                          const std::vector<unsigned> &shape,
                          const T *hostBuff,
                          int bank,
                          int axiWidth) {
  SetPlatform(PLATFORMS::XIL);
  T *paddedHostBuff = PadHostBuffer(shape,hostBuff,axiWidth);
  CloneFrom(xilInfo,GetShape(),paddedHostBuff,bank,axiWidth,CL_BLOCKING);
  delete[](paddedHostBuff);
}

template<typename T>
int CTensorXil<T>::GetDramBank() const {
  return m_iDramBank;
}
template<typename T>
void CTensorXil<T>::SetTensorTag(std::string &tag) {
  m_strTensorTag = tag;
}
template<typename T>
std::string CTensorXil<T>::GetTensorTag() const {
  return m_strTensorTag;
}
template<typename T>
std::vector<unsigned> CTensorXil<T>::GetShapePadded() const {
  return PadShape(GetShape(), m_iAxiWidth);
}
template<typename T>
unsigned CTensorXil<T>::GetLenPadded() const {
  auto paddedShape = GetShapePadded();
  return std::accumulate(begin(paddedShape), end(paddedShape), 1, std::multiplies<unsigned>());
}
template<typename T>
unsigned CTensorXil<T>::GetSizeBytesPadded() const {
  return GetLenPadded() * sizeof(T);
}
template<typename T>
unsigned CTensorXil<T>::GetVectorCountPadded() const {
  return GetLenPadded()/m_iAxiWidth;
}
template<typename T>
cl::Event* CTensorXil<T>::GetEventPtr() {
  return &m_oEvent;
}

template<typename T>
void CTensorXil<T>::CloneFrom(const CTensorXil<T> &other) {
  if (this != &other) { // not a self-assignment
    m_ptrCallBackData.reset(new CallbackData());
    m_ptrXilInfo = other.GetXilInfo(); // shallow-copy, there should be only one copy of CXilInfo and
    // it should be managed by Xilinx implementation class.
    m_iAxiWidth = other.GetAxiWidth();
    m_iDramBank = other.GetDramBank();
    m_strTensorTag = other.GetTensorTag();
    SetPlatform(PLATFORMS::XIL);
    SetShape(other.GetShape());

    cl_mem_ext_ptr_t extPtr = CreateExtendedPointer(nullptr, TranslateBankIndex(m_iDramBank));
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    //flags |= CL_MEM_USE_HOST_PTR;
    flags |= CL_MEM_EXT_PTR_XILINX;

    OclCheck(m_iOclStatus,
             m_oDeviceBuffer =
                 cl::Buffer(*m_ptrXilInfo->GetContext(), flags, GetSizeBytesPadded(), &extPtr, &m_iOclStatus)
    );
    OclCheck(m_iOclStatus,
             m_iOclStatus = m_ptrXilInfo->GetQueue()->enqueueCopyBuffer(
                 other.GetDeviceBuffer(),
                 GetDeviceBuffer(),
                 0,
                 0,
                 GetSizeBytesPadded(),
                 {other.GetEventPtr()},
                 GetEventPtr()
             )
    );
  }
}
template<typename T>
void CTensorXil<T>::CloneFrom(CXilinxInfo *xilInfo,
                              const std::vector<unsigned> &shape,
                              const T *hostBuff,
                              int bank,
                              int axiWidth,
                              cl_bool isBlocking) {
  // WARNING:
  //   THIS METHOD IS NOT RESPONSIBLE FOR MAKING SURE THAT `hostBuff` IS NOT
  //   GOING TO GET RELEASED BEFORE NON BLOCKING OPERATION EXECUTES.
  m_ptrCallBackData.reset(new CallbackData());
  ValidateBankIndex(bank);
  m_iDramBank = bank==-1? m_iDramBank : bank;
  m_iAxiWidth = axiWidth;
  m_ptrXilInfo = xilInfo;
  SetShape(shape);

  cl_mem_ext_ptr_t extPtr = CreateExtendedPointer(nullptr, TranslateBankIndex(m_iDramBank));
  cl_mem_flags  flags = CL_MEM_READ_WRITE;
  //flags |= CL_MEM_USE_HOST_PTR;
  flags |= CL_MEM_EXT_PTR_XILINX;

  OclCheck(m_iOclStatus,
           m_oDeviceBuffer = cl::Buffer(*m_ptrXilInfo->GetContext(), flags, GetSizeBytesPadded(), &extPtr, &m_iOclStatus)
  );

  OclCheck(m_iOclStatus,
           m_iOclStatus = m_ptrXilInfo->GetQueue()->enqueueWriteBuffer(
               m_oDeviceBuffer,
               isBlocking,
               0,
               GetSizeBytesPadded(),
               hostBuff,
               nullptr,
               &m_oEvent)
  );
}
template<typename T>
void CTensorXil<T>::CloneFrom(CXilinxInfo *xilInfo,
                              const std::vector<unsigned> &shape,
                              int bank,
                              int axiWidth) {
  m_ptrCallBackData.reset(new CallbackData());
  ValidateBankIndex(bank);
  m_iDramBank = bank==-1? m_iDramBank : bank;
  m_iAxiWidth = axiWidth;
  m_ptrXilInfo = xilInfo;
  SetShape(shape);

  cl_mem_ext_ptr_t extPtr = CreateExtendedPointer(nullptr, TranslateBankIndex(m_iDramBank));
  cl_mem_flags  flags = CL_MEM_READ_WRITE;
  //flags |= CL_MEM_USE_HOST_PTR;
  flags |= CL_MEM_EXT_PTR_XILINX;

  OclCheck(m_iOclStatus,
           m_oDeviceBuffer = cl::Buffer(*m_ptrXilInfo->GetContext(), flags, GetSizeBytesPadded(), &extPtr, &m_iOclStatus)
  );
}

/*!
 * Pads the data of `hostTn` (not in-place) according to `axiWidth` in order for the device side buffer to
 * comply with the padded last dim policy.
 * host-device transfers are blocking.
 * @tparam T
 * @param context
 * @param queue
 * @param hostTn
 * @param bank
 * @param axiWidth
 */
template<typename T>
CTensorXil<T>::CTensorXil(CXilinxInfo *xilInfo,
                          const CTensor<T> &hostTn,
                          int bank,
                          int axiWidth) {
  SetPlatform(PLATFORMS::XIL);
  T *paddedHostBuff = PadHostBuffer(hostTn.GetShape(),hostTn.GetConst(),axiWidth);
  CloneFrom(xilInfo,hostTn.GetShape(),paddedHostBuff,bank,axiWidth,CL_BLOCKING);
  delete[](paddedHostBuff);
}
template<typename T>
CTensor<T> *CTensorXil<T>::TransferToHost() {

  T *paddedHostBuff = new T[GetLenPadded()];
  std::vector<cl::Event> dependencies;
  dependencies.push_back(m_oEvent);
  OclCheck(m_iOclStatus,
           m_iOclStatus = m_ptrXilInfo->GetQueue()->enqueueReadBuffer(
               m_oDeviceBuffer,
               CL_BLOCKING,
               0,
               GetSizeBytesPadded(),
               paddedHostBuff,
               &dependencies,
               NULL)
  );
  ///TODO enforce queue to be flushed and wait for it to happen?
  ///     OR
  ///     wait on the event to happen?!

  auto *unpaddedHostTn = UnPadHostBuffer(GetShape(), paddedHostBuff, m_iAxiWidth);
  auto *hostTn = new CTensor<T>(GetShape(),unpaddedHostTn);
  delete[](unpaddedHostTn);
  return hostTn;
}
template<typename T>
T *CTensorXil<T>::PadHostBuffer(const std::vector<unsigned> &actualShape, const T *hostSrcBuff, int axiWidth) {
  std::vector<unsigned> paddedShape = PadShape(actualShape, axiWidth);
  unsigned paddedLen = std::accumulate(begin(paddedShape), end(paddedShape), 1, std::multiplies<unsigned>());

  const unsigned sliceCount = paddedLen / paddedShape[paddedShape.size()-1];
  const int actualSliceLen = actualShape[actualShape.size()-1];
  const int paddedSliceLen = paddedShape[actualShape.size()-1];
  T *paddedBuff = new T[paddedLen];

  for(unsigned slice=0; slice<sliceCount; slice++){
    for(int i=0; i<paddedSliceLen; i++){
      paddedBuff[slice*paddedSliceLen + i] = (i<actualSliceLen)? hostSrcBuff[slice*actualSliceLen + i] : 0;
    }
  }

  return paddedBuff;
}

template<typename T>
std::vector<unsigned> CTensorXil<T>::PadShape(const std::vector<unsigned> &shape, int axiWidth) const {
  auto paddedShape = shape;
  // always pad the last dimension.
  unsigned lastDim = paddedShape[paddedShape.size()-1];
  paddedShape[paddedShape.size()-1] = MakeDivisible<unsigned>(lastDim, axiWidth);

  return paddedShape;
}
template<typename T>
T *CTensorXil<T>::UnPadHostBuffer(const std::vector<unsigned> &actualShape, const T *hostSrcBuff, int axiWidth) {
  std::vector<unsigned> paddedShape = PadShape(actualShape, axiWidth);
  unsigned paddedLen = 1;
  for(int i=0; i<paddedShape.size(); i++){
    paddedLen = paddedLen * paddedShape[i];
  }

  const unsigned sliceCount = paddedLen / paddedShape[paddedShape.size()-1];
  const int actualSliceLen = actualShape[actualShape.size()-1];
  const int paddedSliceLen = paddedShape[actualShape.size()-1];
  T *unpaddedBuff = new T[paddedLen];

  for(unsigned slice=0; slice<sliceCount; slice++){
    for(int i=0; i<actualSliceLen; i++){
      unpaddedBuff[slice*actualSliceLen + i] = hostSrcBuff[slice*paddedSliceLen + i];
    }
  }

  return unpaddedBuff;
}

template<typename T>
CTensorXil<T>* CTensorXil<T>::CloneIfNeededToBank(const unsigned destBank) {
  if(m_iDramBank==destBank) return this;

  if(!(m_iDramBank>=0 && m_iDramBank<=3)){
    throw std::runtime_error(CStringFormatter()<< __func__ << ": Invalid or unsupported tensor src bank.");
    //SPDLOG_LOGGER_ERROR(logger,"Invalid or unsupported srcBank");
    //std::exit(1);
  }
  if(!(destBank>=0 && destBank<=3)){
    throw std::runtime_error(CStringFormatter()<< __func__ << ": Invalid or unsupported tensor dest bank.");
    //SPDLOG_LOGGER_ERROR(logger,"Invalid or unsupported dstBank");
    //std::exit(1);
  }

  auto *newTensor = new CTensorXil(m_ptrXilInfo, GetShape(),false,destBank,GetAxiWidth());

  int argcnt=0;
  // arguments should be like: bank0 only, bank1 only, bank2 only, and bank3 only.

  unsigned vecCountPadded = GetVectorCountPadded();

  //Bank0
#ifdef USEMEMORYBANK0
  if(m_iDramBank==0 || destBank==0){
    if(m_iDramBank==0){
      OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, m_oDeviceBuffer));
    }else{
      OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, newTensor->GetDeviceBuffer()));
    }
  }else{
    if(GetLen() > m_ptrXilInfo->GetDatamoverDummyTensor(0)->GetLen()){
      throw std::runtime_error(CStringFormatter()<< __func__ << ": Increase DummyDatamoverTensor sizes, the min len should be "<< GetLen());
    }
    OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, ((CTensorXil<float>*)m_ptrXilInfo->GetDatamoverDummyTensor(0))->GetDeviceBuffer()));
  }
#endif

  //Bank1
#ifdef USEMEMORYBANK1
  if(m_iDramBank==1 || destBank==1){
    if(m_iDramBank==1){
      OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, m_oDeviceBuffer));
    }else{
      OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, newTensor->GetDeviceBuffer()));
    }
  }else{
    if(GetLen() > m_ptrXilInfo->GetDatamoverDummyTensor(1)->GetLen()){
      throw std::runtime_error(CStringFormatter()<< __func__ << ": Increase DummyDatamoverTensor sizes, the min len should be "<< GetLen());
    }
    OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, ((CTensorXil<float>*)m_ptrXilInfo->GetDatamoverDummyTensor(1))->GetDeviceBuffer()));
  }
#endif

  //Bank2
#ifdef USEMEMORYBANK2
  if(m_iDramBank==2 || destBank==2){
        if(m_iDramBank==2){
            OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, m_oDeviceBuffer));
        }else{
            OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, newTensor->GetDeviceBuffer()));
        }
    }else{
        if(GetLen() > m_ptrXilInfo->GetDatamoverDummyTensor(2)->GetLen()){
          throw std::runtime_error(CStringFormatter()<< __func__ << ": Increase DummyDatamoverTensor sizes, the min len should be "<< GetLen());
        }
        OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, ((CTensorXil<float>*)m_ptrXilInfo->GetDatamoverDummyTensor(2))->GetDeviceBuffer()));
    }
#endif

  //Bank3
#ifdef USEMEMORYBANK3
  if(m_iDramBank==3 || destBank==3){
        if(m_iDramBank==3){
            OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, m_oDeviceBuffer));
        }else{
            OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, newTensor->GetDeviceBuffer()));
        }
    }else{
        if(GetLen() > m_ptrXilInfo->GetDatamoverDummyTensor(3)->GetLen()){
          throw std::runtime_error(CStringFormatter()<< __func__ << ": Increase DummyDatamoverTensor sizes, the min len should be "<< GetLen());
        }
        OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, ((CTensorXil<float>*)m_ptrXilInfo->GetDatamoverDummyTensor(3))->GetDeviceBuffer()));
    }
#endif

  OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, m_iDramBank));
  OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, destBank));
  OclCheck(m_iOclStatus, m_iOclStatus = m_ptrXilInfo->GetDatamoverKernel()->setArg(argcnt++, vecCountPadded));

  std::vector<cl::Event> dependencies;
  dependencies.push_back(m_oEvent);

  OclCheck(m_iOclStatus,
      m_iOclStatus = m_ptrXilInfo->GetQueue()->enqueueTask(*(m_ptrXilInfo->GetDatamoverKernel()), &dependencies, newTensor->GetEventPtr())
  );


  m_ptrCallBackData.get()->profileKernel = m_ptrXilInfo->GetProfileOclEnabled();
  m_ptrCallBackData.get()->parentLayerId = 4294967295;
  m_ptrCallBackData.get()->classPtr = m_ptrXilInfo;
  newTensor->GetEventPtr()->setCallback(CL_COMPLETE, &EventCallback, m_ptrCallBackData.get());

  return newTensor;
}

template<typename T>
CXilinxInfo *CTensorXil<T>::GetXilInfo() const {
  return m_ptrXilInfo;
}
template<typename T>
unsigned long CTensorXil<T>::GetSizeBytes() const {
  return GetLen()*sizeof(T);
}





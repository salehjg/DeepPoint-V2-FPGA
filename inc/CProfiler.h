#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "xilinx/config.h"
#include "GlobalHelpers.h"

class CProfiler{
 public:
  using DictShapePtr = std::unordered_map<std::string, std::vector<unsigned>>;
  using DictIntPtr = std::unordered_map<std::string, int>;
  using DictFloatPtr = std::unordered_map<std::string, float>;

  CProfiler(const std::string &fnameJson, bool enableCpuUsageSampling);

  ~CProfiler();

  void StartLayer(PLATFORMS platform,
                  const unsigned layerId,
                  const std::string &name,
                  DictShapePtr *dictShapes,
                  DictIntPtr *dictScalarInt,
                  DictFloatPtr *dictScalarFloat);

  void FinishLayer();

  void StartKernel(PLATFORMS platform,
                   const unsigned parentLayerId,
                   const std::string &name,
                   const unsigned long durationNanoSeconds);

  void StartKernelDatamover(PLATFORMS platform,
                   const unsigned parentLayerId,
                   const unsigned vecCountPadded,
                   const std::string &name,
                   const unsigned long durationNanoSeconds);

  void FinishKernel();
  float GetLastCpuUsage();

 private:
  long GetTimestampMicroseconds();
  float _GetCpuUsage();
  void CpuUsageThread();

  rapidjson::StringBuffer m_oStrBuffer;
  rapidjson::Writer<rapidjson::StringBuffer> *m_ptrWriter;
  std::ofstream *m_ptrFileStream;
  std::string m_strFileName;
  bool m_bEnableCpuUsageSampling;

  std::atomic<float> m_fCpuUsage;
  std::atomic<bool> m_bStopThread;
  std::thread m_oThread;
  unsigned long long m_lLastTotalUser, m_lLastTotalUserLow, m_lLastTotalSys, m_lLastTotalIdle;
};

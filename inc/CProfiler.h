#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
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

  CProfiler(const std::string &fnameJson);

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

  void FinishKernel();


 private:
  long GetTimestampMicroseconds();
  rapidjson::StringBuffer m_oStrBuffer;
  rapidjson::Writer<rapidjson::StringBuffer> *m_ptrWriter;
  std::ofstream *m_ptrFileStream;

};

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

class CProfiler{
 public:
  using DictShapePtr = std::unordered_map<std::string, std::vector<unsigned>>;
  using DictIntPtr = std::unordered_map<std::string, int>;
  using DictFloatPtr = std::unordered_map<std::string, float>;

  CProfiler(const std::string &fnameJson="output.json");

  ~CProfiler();

  void StartLayer(
      unsigned time,
      const std::string &name,
      DictShapePtr *dictShapes,
      DictIntPtr *dictScalarInt,
      DictFloatPtr *dictScalarFloat);

  void FinishLayer(unsigned time);

  void StartKernel(unsigned time,
                   const std::string &name,
                   DictShapePtr *dictShapes,
                   DictIntPtr *dictScalarInt,
                   DictFloatPtr *dictScalarFloat);

  void FinishKernel(unsigned time);
  long GetTimestampMicroseconds();

 private:
  rapidjson::StringBuffer m_oStrBuffer;
  rapidjson::Writer<rapidjson::StringBuffer> *m_ptrWriter;
  std::ofstream *m_ptrFileStream;

};

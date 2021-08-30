#include "CProfiler.h"
#include <chrono>



CProfiler::CProfiler(const std::string &fnameJson) {
  m_ptrWriter = new rapidjson::Writer<rapidjson::StringBuffer>(m_oStrBuffer);
  m_ptrFileStream = new std::ofstream(fnameJson);
  m_ptrWriter->StartObject();

  m_ptrWriter->Key("info");
  m_ptrWriter->StartObject();
  {
    m_ptrWriter->Key("commit.host");
    m_ptrWriter->String(REPO_HASH_MAIN);
    m_ptrWriter->Key("commit.config");
    m_ptrWriter->String(REPO_HASH_CONFIG);
  }
  m_ptrWriter->EndObject();

  m_ptrWriter->Key("trace");
  m_ptrWriter->StartArray();

}

CProfiler::~CProfiler() {
  m_ptrWriter->EndArray(); // main trace array end.
  m_ptrWriter->EndObject(); // main json end.
  *m_ptrFileStream<<m_oStrBuffer.GetString();
  m_ptrFileStream->close();
  delete m_ptrFileStream;
  delete m_ptrWriter;
}

void CProfiler::StartLayer(PLATFORMS platform,
                           const unsigned layerId,
                           const std::string &name,
                           CProfiler::DictShapePtr *dictShapes,
                           CProfiler::DictIntPtr *dictScalarInt,
                           CProfiler::DictFloatPtr *dictScalarFloat) {
  m_ptrWriter->StartObject();
  m_ptrWriter->Key("type");
  m_ptrWriter->String("layer");
  m_ptrWriter->Key("platform");
  m_ptrWriter->String(platform==PLATFORMS::CPU? "cpu": (platform==PLATFORMS::XIL? "xil": "undef"));
  m_ptrWriter->Key("id");
  m_ptrWriter->Uint(layerId);
  m_ptrWriter->Key("args");
  m_ptrWriter->StartObject();
  {
    for(auto &item: *dictShapes){
      m_ptrWriter->Key(item.first.c_str());
      m_ptrWriter->StartArray();
      for(auto &dim: item.second){
        m_ptrWriter->Uint(dim);
      }
      m_ptrWriter->EndArray();
    }

    for(auto &item: *dictScalarInt){
      m_ptrWriter->Key(item.first.c_str());
      m_ptrWriter->Int(item.second);
    }

    for(auto &item: *dictScalarFloat){
      m_ptrWriter->Key(item.first.c_str());
      m_ptrWriter->Double((double)item.second);
    }
  }
  m_ptrWriter->EndObject();
  m_ptrWriter->Key("time.start");
  m_ptrWriter->Uint64(GetTimestampMicroseconds());
  m_ptrWriter->Key("nested");
  m_ptrWriter->StartArray();
}

void CProfiler::FinishLayer() {
  m_ptrWriter->EndArray();
  m_ptrWriter->Key("time.stop");
  m_ptrWriter->Uint64(GetTimestampMicroseconds());
  m_ptrWriter->EndObject();
}

void CProfiler::StartKernel(PLATFORMS platform,
                            const unsigned parentLayerId,
                            const std::string &name,
                            const unsigned long durationNanoSeconds) {
  m_ptrWriter->StartObject();
  m_ptrWriter->Key("type");
  m_ptrWriter->String("kernel");
  m_ptrWriter->Key("platform");
  m_ptrWriter->String(platform==PLATFORMS::CPU? "cpu": (platform==PLATFORMS::XIL? "xil": "undef"));
  m_ptrWriter->Key("id");
  m_ptrWriter->Uint(parentLayerId);
  m_ptrWriter->Key("duration");
  m_ptrWriter->Uint64(durationNanoSeconds);
}

void CProfiler::FinishKernel() {
  m_ptrWriter->EndObject();
}

long CProfiler::GetTimestampMicroseconds() {
  auto now = std::chrono::steady_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::microseconds>(now);
  auto epoch = now_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::microseconds>(epoch);
  return value.count();
}

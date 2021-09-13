#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "CProfiler.h"
#include <chrono>

CProfiler::CProfiler(const std::string &fnameJson) {
  m_strFileName = fnameJson;
  m_ptrWriter = new rapidjson::Writer<rapidjson::StringBuffer>(m_oStrBuffer);
  m_ptrFileStream = new std::ofstream(m_strFileName);
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


  SPDLOG_LOGGER_TRACE(logger, "Spinning CProfiler's thread to poll the CPU usage.");
  m_bStopThread = false;
  FILE* file = fopen("/proc/stat", "r");
  fscanf(file, "cpu %llu %llu %llu %llu",
      &m_lLastTotalUser,
      &m_lLastTotalUserLow,
      &m_lLastTotalSys,
      &m_lLastTotalIdle);
  fclose(file);
  std::thread th(&CProfiler::CpuUsageThread, this);
  swap(th, m_oThread);
  SPDLOG_LOGGER_TRACE(logger, "Done.");
}

CProfiler::~CProfiler() {
  SPDLOG_LOGGER_TRACE(logger, "Writing profiling data to {}", m_strFileName);
  m_ptrWriter->EndArray(); // main trace array end.
  m_ptrWriter->EndObject(); // main json end.
  *m_ptrFileStream<<m_oStrBuffer.GetString();
  m_ptrFileStream->close();

  SPDLOG_LOGGER_TRACE(logger, "Stopping CProfiler's CPU usage polling thread.");
  m_bStopThread = true;
  m_oThread.join();
  SPDLOG_LOGGER_TRACE(logger, "Done.");

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
  m_ptrWriter->Key("name");
  m_ptrWriter->String(name.c_str());
  m_ptrWriter->Key("platform");
  m_ptrWriter->String(platform==PLATFORMS::CPU? "cpu": (platform==PLATFORMS::XIL? "xil": "undef"));
  m_ptrWriter->Key("id");
  m_ptrWriter->Uint(layerId);
  m_ptrWriter->Key("args");
  m_ptrWriter->StartObject();
  {
    if(dictShapes!= nullptr){
      for(auto &item: *dictShapes){
        m_ptrWriter->Key(item.first.c_str());
        m_ptrWriter->StartArray();
        for(auto &dim: item.second){
          m_ptrWriter->Uint(dim);
        }
        m_ptrWriter->EndArray();
      }
    }

    if(dictScalarInt!= nullptr) {
      for (auto &item: *dictScalarInt) {
        m_ptrWriter->Key(item.first.c_str());
        m_ptrWriter->Int(item.second);
      }
    }

    if(dictScalarFloat!= nullptr) {
      for (auto &item: *dictScalarFloat) {
        m_ptrWriter->Key(item.first.c_str());
        m_ptrWriter->Double((double) item.second);
      }
    }
  }
  m_ptrWriter->EndObject();
  m_ptrWriter->Key("time.start");
  m_ptrWriter->Uint64(GetTimestampMicroseconds());
  m_ptrWriter->Key("cpu.usage");
  m_ptrWriter->Double(GetLastCpuUsage());
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
  m_ptrWriter->Key("name");
  m_ptrWriter->String(name.c_str());
  m_ptrWriter->Key("platform");
  m_ptrWriter->String(platform==PLATFORMS::CPU? "cpu": (platform==PLATFORMS::XIL? "xil": "undef"));
  m_ptrWriter->Key("id");
  m_ptrWriter->Uint(parentLayerId);
  m_ptrWriter->Key("duration");
  m_ptrWriter->Uint64(durationNanoSeconds);
}

void CProfiler::StartKernelDatamover(PLATFORMS platform,
                            const unsigned parentLayerId,
                            const unsigned vecCountPadded,
                            const std::string &name,
                            const unsigned long durationNanoSeconds) {
  m_ptrWriter->StartObject();
  m_ptrWriter->Key("type");
  m_ptrWriter->String("kernel");
  m_ptrWriter->Key("name");
  m_ptrWriter->String(name.c_str());
  m_ptrWriter->Key("platform");
  m_ptrWriter->String(platform==PLATFORMS::CPU? "cpu": (platform==PLATFORMS::XIL? "xil": "undef"));
  m_ptrWriter->Key("id");
  m_ptrWriter->Uint(parentLayerId);
  m_ptrWriter->Key("bytes");
  m_ptrWriter->Uint(vecCountPadded*CONFIG_M_AXI_WIDTH*CONFIG_DTYPE_SIZE);
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

float CProfiler::_GetCpuUsage() {
  float percent;
  FILE* file;
  unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;

  file = fopen("/proc/stat", "r");
  fscanf(file, "cpu %llu %llu %llu %llu",
      &totalUser,
      &totalUserLow,
      &totalSys,
      &totalIdle);
  fclose(file);

  if( totalUser < m_lLastTotalUser ||
      totalUserLow < m_lLastTotalUserLow ||
      totalSys < m_lLastTotalSys ||
      totalIdle < m_lLastTotalIdle){
    //Overflow detection. Just skip this value.
    percent = -1.0f;
  }else{
    total =
        (totalUser - m_lLastTotalUser) +
        (totalUserLow - m_lLastTotalUserLow) +
        (totalSys - m_lLastTotalSys);
    percent = (float)total;
    total += (totalIdle - m_lLastTotalIdle);
    percent /= (float)total;
    percent *= 100.0f;
  }

  m_lLastTotalUser = totalUser;
  m_lLastTotalUserLow = totalUserLow;
  m_lLastTotalSys = totalSys;
  m_lLastTotalIdle = totalIdle;

  return percent;
}

void CProfiler::CpuUsageThread() {
  while (!m_bStopThread) {
    m_fCpuUsage = _GetCpuUsage();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

float CProfiler::GetLastCpuUsage(){
  return m_fCpuUsage;
}

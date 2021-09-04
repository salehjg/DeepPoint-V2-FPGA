#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "argparse.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"
#include "GlobalHelpers.h"

using namespace std;
using namespace argparse;

spdlog::logger *logger;
string globalArgXclBin;
string globalArgDataPath;
unsigned globalBatchsize;
bool globalDumpTensors=false;
bool globalProfileOclEnabled=true;
bool globalModelnet=true;
bool globalShapenet=false;

void Handler(int sig) {
  void *array[40];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 40);

  // print out all the frames to stderr
  cerr<<"The host program has crashed, printing call stack:\n";
  cerr<<"Error: signal "<< sig<<"\n";
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  SPDLOG_LOGGER_CRITICAL(logger,"The host program has crashed.");
  spdlog::shutdown();

  exit(SIGSEGV);
}

void HandlerInt(int sig_no)
{
  SPDLOG_LOGGER_CRITICAL(logger,"CTRL+C pressed, terminating...");
  spdlog::shutdown();
  exit(SIGINT);
}

void SetupModules(int argc, const char* argv[]){
  signal(SIGSEGV, Handler);
  signal(SIGABRT, Handler);
  signal(SIGINT, HandlerInt);

  ArgumentParser parser(argv[0], "DeeppointV2FPGA");
  parser.add_argument()
      .names({"-i", "--image"})
      .description("FPGA image(*.xclbin or *.awsxclbin)")
      .required(true);

  parser.add_argument()
      .names({"-d", "--data"})
      .description("Data directory")
      .required(true);

  parser.add_argument()
      .names({"-y", "--shapenet2"})
      .description("Use ShapeNetV2 dataset instead of ModelNet40 (no value is needed for this argument)")
      .required(false);

  parser.add_argument()
      .names({"-b", "--batchsize"})
      .description("Batch-size")
      .required(false);

  parser.add_argument()
      .names({"-e", "--emumode"})
      .description("Forced emulation mode (sw_emu or hw_emu)")
      .required(false);

  parser.add_argument()
      .names({"-k", "--dumptensors"})
      .description("Dump tensors into *.npy files in the data directory (no value is needed for this argument)")
      .required(false);

  parser.add_argument()
      .names({"-n", "--nolog"})
      .description("Disable logging (no value is needed for this argument)")
      .required(false);

  parser.add_argument()
      .names({"--noprofileocl"})
      .description("Disable OpenCL profiling to increase performance. (Enabled by default)(no value is needed for this argument)")
      .required(false);

  parser.enable_help();
  auto err = parser.parse(argc, argv);
  if(err){
    std::cerr << err << std::endl;
    parser.print_help();
    exit(EXIT_FAILURE);
  }

  if(parser.exists("help")){
    parser.print_help();
    exit(EXIT_SUCCESS);
  }

  {
    // HOST LOGGER
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
    console_sink->set_pattern("[%H:%M:%S.%e][%^%l%$] %v");

    auto file_sink1 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("hostlog_0trace.log", true);
    file_sink1->set_level(spdlog::level::trace);
    file_sink1->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");


    auto file_sink3 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("hostlog_3wraning.log", true);
    file_sink3->set_level(spdlog::level::warn);
    file_sink3->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");

    logger = new spdlog::logger("DP2FPGA Host-logger", {console_sink, file_sink1, file_sink3});
    logger->set_level(spdlog::level::trace);

    if(parser.exists("n")) {
      logger->set_level(spdlog::level::off);
    }
    //SPDLOG_LOGGER_TRACE(logger,"test log ::: trace");
    //SPDLOG_LOGGER_DEBUG(logger,"test log ::: debug");
    //SPDLOG_LOGGER_INFO(logger,"test log ::: info");
    //SPDLOG_LOGGER_WARN(logger,"test log ::: warn");
    //SPDLOG_LOGGER_ERROR(logger,"test log ::: error");
    //SPDLOG_LOGGER_CRITICAL(logger,"test log ::: critical");
  }

  if(parser.exists("b")) {
    globalBatchsize = parser.get<unsigned>("b");
  }else{
    globalBatchsize = 5;
  }
  SPDLOG_LOGGER_INFO(logger,"Batch-size: {}", globalBatchsize);

  if(parser.exists("i")) {
    globalArgXclBin = parser.get<string>("i");
    SPDLOG_LOGGER_INFO(logger,"FPGA Image: {}", globalArgXclBin);
  }

  if(parser.exists("d")) {
    globalArgDataPath = parser.get<string>("d");
    SPDLOG_LOGGER_INFO(logger,"Data Directory: {}", globalArgDataPath);
  }

  if(parser.exists("e")) {
    const char *forcedMode = parser.get<string>("e").c_str();
    SPDLOG_LOGGER_INFO(logger,"Forced Emulation Mode: {}", forcedMode);
    if (setenv("XCL_EMULATION_MODE", forcedMode, 1) < 0) {
      std::cerr <<""<<std::endl;
      SPDLOG_LOGGER_ERROR(logger,"Can not set env var XCL_MODE.");
    }
  }

  if(parser.exists("x") && parser.exists("y")) {
    std::cerr <<"Both datasets cannot be used at the same time!"<<std::endl;
    SPDLOG_LOGGER_ERROR(logger,"Both datasets cannot be used at the same time!");
    exit(EXIT_FAILURE);
  }

  if(parser.exists("y")) {
    SPDLOG_LOGGER_INFO(logger,"The selected dataset is ShapeNetV2.");
    globalModelnet=false;
    globalShapenet=true;
  }else{
    SPDLOG_LOGGER_INFO(logger,"The selected dataset is ModelNet40.");
    globalModelnet=true;
    globalShapenet=false;
  }

  if(parser.exists("dumptensors")) {
    globalDumpTensors = true;
    SPDLOG_LOGGER_INFO(logger,"Tensors will be dumped into separate numpy files in the data directory.");
  }

  if(parser.exists("noprofileocl")) {
    globalProfileOclEnabled = false;
    SPDLOG_LOGGER_INFO(logger,"The OpenCL profiling is forcibly disabled to increase performance.");
  } else{
    SPDLOG_LOGGER_WARN(logger,"The OpenCL profiling is enabled and is going to impose some serious host-side overhead.");
  }
}

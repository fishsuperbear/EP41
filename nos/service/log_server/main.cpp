#include <signal.h>
#include <thread>
#include <fstream>

#include "json/json.h"
#include "log/include/logging.h"
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"

#include "log_server/log/log_server_logger.h"
#include "log_server/manager/log_server.h"

#define LOG_SERVER_CONFIG_FILE_FOR_MDC_LLVM   ("/opt/usr/diag_update/mdc-llvm/conf/logserver_config.json")
#define LOG_SERVER_CONFIG_FILE_FOR_J5         ("/userdata/diag_update/j5/conf/logserver_config.json")
#define LOG_SERVER_CONFIG_FILE_FOR_ORIN       ("/app/runtime_service/log_server/conf/logserver_config.json")
#define LOG_SERVER_CONFIG_FILE_DEFAULT        ("/app/runtime_service/log_server/conf/logserver_config.json")

using namespace hozon::netaos::logserver;
using namespace hozon::netaos::em;
int g_signum = 0;
LogServer ls;
std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
void SigHandler(int signum)
{
    g_signum = signum;
    execli->ReportState(ExecutionState::kTerminating);
    ls.Stop();
}

void InitLog()
{
    std::string configFile;
    std::string LogAppName = "LOGSV";
    std::string LogAppDescription = "log_server";
    std::string LogContextName = "LOGSV";
    std::string LogFilePath = "/opt/usr/log/soc_log/";
    uint32_t    LogLevel = 1;  // kDebug
    uint32_t    LogMode = 2;   // 1: console 2: File 3: console&file
    uint32_t    MaxLogFileNum = 10;
    uint32_t    MaxSizeOfLogFile = 20;

#ifdef BUILD_FOR_MDC
    configFile = LOG_SERVER_CONFIG_FILE_FOR_MDC_LLVM;
#elif BUILD_FOR_J5
    configFile = LOG_SERVER_CONFIG_FILE_FOR_J5;
#elif DBUILD_FOR_ORIN
    configFile = LOG_SERVER_CONFIG_FILE_FOR_ORIN;
#else
    configFile = LOG_SERVER_CONFIG_FILE_DEFAULT;
#endif

    if (0 == access(configFile.c_str(), F_OK)) {
        Json::Value rootReder;
        Json::CharReaderBuilder readBuilder;
        std::ifstream ifs(configFile);
        std::unique_ptr<Json::CharReader> reader(readBuilder.newCharReader());
        JSONCPP_STRING errs;
        if (Json::parseFromStream(readBuilder, ifs, &rootReder, &errs)) {
            LogAppName = (rootReder["LogAppName"]) ? rootReder["LogAppName"].asString() : LogAppName;
            LogAppDescription = (rootReder["LogAppDescription"]) ? rootReder["LogAppDescription"].asString() : LogAppDescription;
            LogContextName = (rootReder["LogContextName"]) ? rootReder["LogContextName"].asString() : LogContextName;
            LogFilePath = (rootReder["LogFilePath"]) ? rootReder["LogFilePath"].asString() : LogFilePath;
            LogLevel = (rootReder["LogLevel"]) ? rootReder["LogLevel"].asUInt() : LogLevel;
            LogMode = (rootReder["LogMode"]) ? rootReder["LogMode"].asUInt() : LogMode;
            MaxLogFileNum = (rootReder["MaxLogFileNum"]) ? rootReder["MaxLogFileNum"].asUInt() : MaxLogFileNum;
            MaxSizeOfLogFile = (rootReder["MaxSizeOfLogFile"]) ? rootReder["MaxSizeOfLogFile"].asUInt() : MaxSizeOfLogFile;
        }
    }

    LogServerLogger::GetInstance().InitLogging(LogAppName,    // the id of application
        LogAppDescription, // the log id of application
        static_cast<hozon::netaos::log::LogLevel>(LogLevel), //the log level of application
        LogMode, //the output log mode
        LogFilePath, //the log file directory, active when output log to file
        MaxLogFileNum, //the max number log file , active when output log to file
        MaxSizeOfLogFile //the max size of each  log file , active when output log to file
    );
    LogServerLogger::GetInstance().CreateLogger(LogContextName);
}


int main(int argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    InitLog();

    execli->ReportState(ExecutionState::kRunning);

    ls.Init();
    ls.Run();

    return 0;
}
#include <signal.h>
#include <thread>
#include <fstream>
#include "json/json.h"

#include "update_manager/manager/update_manager.h"
#include "log/update_manager_logger.h"
#include "log/include/logging.h"
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"

#define UPDATE_CONFIG_FILE_FOR_MDC   ("/opt/usr/diag_update/mdc-llvm/conf/update_config.json")
#define UPDATE_CONFIG_FILE_FOR_J5    ("/userdata/diag_update/j5/conf/update_config.json")
#define UPDATE_CONFIG_FILE_ORIN      ("/app/runtime_service/update_manager/conf/update_config.json")
#define UPDATE_CONFIG_FILE_DEFAULT   ("/app/runtime_service/update_manager/conf/update_config.json")

using namespace hozon::netaos::update;
using namespace hozon::netaos::em;

bool stop_flag = false;
int g_signum = 0;

void SigHandler(int signum)
{
    g_signum = signum;
    stop_flag = true;
}

void InitLog()
{
    std::string configFile;
    std::string LogAppName = "UPMG";
    std::string LogAppDescription = "update_manager";
    std::string LogContextName = "UPMG";
    std::string LogFilePath = "/opt/usr/log/soc_log/";
    uint32_t    LogLevel = 1;  // kDebug
    uint32_t    LogMode = 3;   // Console and File
    uint32_t    MaxLogFileNum = 10;
    uint32_t    MaxSizeOfLogFile = 20;

#ifdef BUILD_FOR_MDC
    configFile = UPDATE_CONFIG_FILE_FOR_MDC;
#elif BUILD_FOR_J5
    configFile = UPDATE_CONFIG_FILE_FOR_J5;
#elif BUILD_FOR_ORIN
    configFile = UPDATE_CONFIG_FILE_ORIN;
#else
    configFile = UPDATE_CONFIG_FILE_DEFAULT;
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

    UpdateManagerLogger::GetInstance().InitLogging(LogAppName,    // the id of application
        LogAppDescription, // the log id of application
        UpdateManagerLogger::UpdateLogLevelType(LogLevel), //the log level of application
        LogMode, //the output log mode
        LogFilePath, //the log file directory, active when output log to file
        MaxLogFileNum, //the max number log file , active when output log to file
        MaxSizeOfLogFile //the max size of each  log file , active when output log to file
    );
    UpdateManagerLogger::GetInstance().CreateLogger(LogContextName);
}


int main(int argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    InitLog();
    UpdateManager um;
    std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
    execli->ReportState(ExecutionState::kRunning);

    um.Init();
    um.Start();

    while (!stop_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    execli->ReportState(ExecutionState::kTerminating);
    um.Stop();
    um.Deinit();

    return 0;
}
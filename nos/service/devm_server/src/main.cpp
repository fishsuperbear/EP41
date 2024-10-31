/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */

#include <signal.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "devm_data_define.h"
#include "devm_server.h"
#include "json/json.h"

#include "em/include/exec_client.h"
#include "em/include/proctypes.h"
#include "devm_server_logger.h"
#include "log/include/default_logger.h"

using namespace hozon::netaos::devm_server;
using namespace hozon::netaos::em;

#define DEVM_SERVER_CONFIG_FILE_DEFAULT ("/app/runtime_service/devm_server/conf/devm_config.json")


sig_atomic_t g_stopFlag = 0;
int g_signum = 0;

void SigHandler(int signum) {
    g_stopFlag = 1;
    g_signum = signum;
    std::cout << "--- cfg SigHandler enter, signum [" << signum << "] ---" << std::endl;
}

static void InitLog() {
    std::string configFile = DEVM_SERVER_CONFIG_FILE_DEFAULT;
    std::string LogAppName = "DEVM";
    std::string LogAppDescription = "devm_server";
    std::string LogContextName = "DEVM";
    std::string LogFilePath = "/opt/usr/log/soc_log/";
    uint32_t LogLevel = 2;  // kInfo
    uint32_t LogMode = 2;   // Console
    uint32_t MaxLogFileNum = 10;
    uint32_t MaxSizeOfLogFile = 20;

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

    DevmServerLogger::GetInstance().InitLogging(LogAppName,                                    // the id of application
                                                LogAppDescription,                             // the log id of application
                                                DevmServerLogger::DEVMLogLevelType(LogLevel),  //the log level of application
                                                LogMode,                                       //the output log mode
                                                LogFilePath,                                   //the log file directory, active when output log to file
                                                MaxLogFileNum,                                 //the max number log file , active when output log to file
                                                MaxSizeOfLogFile                               //the max size of each  log file , active when output log to file
    );
    DevmServerLogger::GetInstance().CreateLogger(LogContextName);
}

int main(int argc, char** argv) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);
    InitLog();
    DEVM_LOG_INFO << "devm_server start...";

    std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
    DevmServer devmserver;
    execli->ReportState(ExecutionState::kRunning);

    devmserver.Init();
    devmserver.Run();
    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    execli->ReportState(ExecutionState::kTerminating);
    devmserver.DeInit();
    DEVM_LOG_INFO << "devm_server end...";

    return 0;
}

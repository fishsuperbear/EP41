/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */
#include <csignal>
#include <thread>
#include <iostream>
#include <chrono>

#include <string>

#include "json/json.h"

#include "em/include/exec_client.h"
#include "em/include/proctypes.h"

#include "network_capture/include/network_capture.h"
#include "network_capture/include/network_logger.h"

/*debug*/#include "network_capture/include/function_statistics.h"
#include "network_capture/include/statistics_define.h"

using namespace hozon::netaos::em;
using namespace hozon::netaos::network_capture;

// static hozon::netaos::network_capture::NetworkCapture network_capture;

sig_atomic_t g_stopFlag = 0;

void SigHandler(int signum)
{

    std::cout << "Received signal: " << signum  << ". Quitting\n";
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    g_stopFlag = true;

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
}

/** Sets up signal handler.*/
void SigSetup(void)
{
    struct sigaction action
    {
    };
    action.sa_handler = SigHandler;

    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGQUIT, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);
}

static void InitLog() {
    std::string configFile = std::string(CONFIG_PATH) + "/common_config.json";
    std::string LogAppName = "CAPTURE";
    std::string LogAppDescription = "network_capture";
    std::string LogContextName = "CAPTURE";
    std::string LogFilePath = "/opt/usr/log/soc_log/";
    uint32_t LogLevel = 2;  // kInfo
    uint32_t LogMode = 2;   // File
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

    NetworkLogger::GetInstance().InitLogging(LogAppName,                                    // the id of application
                                             LogAppDescription,                             // the log id of application
                                             NetworkLogger::NETLogLevelType(LogLevel),      //the log level of application
                                             LogMode,                                       //the output log mode
                                             LogFilePath,                                   //the log file directory, active when output log to file
                                             MaxLogFileNum,                                 //the max number log file , active when output log to file
                                             MaxSizeOfLogFile                               //the max size of each  log file , active when output log to file
    );
    NetworkLogger::GetInstance().CreateLogger(LogContextName);
}

int main(int argc, char* argv[])
{
    SigSetup();
    InitLog();

    NETWORK_LOG_WARN << "network_capture start";
    std::cout << "network_capture start" << std::endl;

    std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
    int32_t ret = execli->ReportState(ExecutionState::kRunning);
    if(ret < 0) {
        NETWORK_LOG_ERROR << "network_capture ExecutionState::kRunning report fail.";
    }
    std::unique_ptr<NetworkCapture> network_capture = std::make_unique<NetworkCapture>();
    // NetworkCapture* network_capture = new NetworkCapture();
    {
    FunctionStatistics("network_capture.Init()");
    network_capture->Init();
    }

    network_capture->Run();
    int64_t statistic_time_last = 0;
    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        {
            std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            std::chrono::seconds seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
            int64_t statistic_time_now = seconds.count();
            int64_t gap = statistic_time_now - statistic_time_last;
            
            if (gap >= 10) {
                Counter::Instance().printStat(gap);
                statistic_time_last = statistic_time_now;
            }
        }
    }

    {
    FunctionStatistics("NW Stop&DeInit & EM report");
    network_capture->Stop();
    network_capture->DeInit();
    }

    ret = execli->ReportState(ExecutionState::kTerminating);
    if(ret < 0) {
        NETWORK_LOG_ERROR << "network_capture ExecutionState::kTerminating report fail.";
    }

    // delete network_capture;
    NETWORK_LOG_WARN << "network_capture stop";
    std::cout << "network_capture stop" << std::endl;
    return 0;
}
/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-10-07 17:47:39
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-12-25 11:59:45
 * @FilePath: /nos/service/config_server/src/main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE#include "include/cfg_server.h"

 */
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: main function definition
 */

#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <ara/core/initialization.h>

#include "em/include/exec_client.h"
#include "em/include/proctypes.h"
#include "include/cfg_server.h"
#include "include/cfg_server_proto.h"
#include "include/phm_client_instance.h"
#include "json/json.h"

using namespace hozon::netaos::em;
using namespace hozon::netaos::cfg;
using namespace hozon::netaos::phm;

sig_atomic_t g_stopFlag = 0;
void SigHandler(int signum) {
    g_stopFlag = 1;
    std::cout << "--- cfg SigHandler enter, signum [" << signum << "] ---" << std::endl;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);
    std::string configFile;
    std::string LogAppName = "CFG_SERVER";
    std::string LogAppDescription = "cfg_server";
    std::string LogContextName = "CFG_SERVER";
    std::string LogFilePath = "/opt/usr/log/soc_log/";
    std::string PerFilePath = "/cfg/";
    std::string BakPerFilePath = "/cfg_bak/";
    uint32_t LogLevel = 2;  // kDebug
    uint32_t LogMode = 2;   // Console and File
    uint32_t MaxLogFileNum = 10;
    uint32_t MaxSizeOfLogFile = 20;
    uint32_t MaxComValLimit = 10240;
    bool cfgparseflag = true;
    bool cfgfoundflag = true;

#ifdef BUILD_FOR_MDC
    configFile = "/opt/usr/config_server/conf/config_server.json";
#elif BUILD_FOR_ORIN
    configFile = "/app/runtime_service/config_server/conf/config_server.json";
#else
    configFile = "./conf/config_server.json";
#endif
    if (0 == access(configFile.c_str(), F_OK)) {
        Json::Value rootReder;
        Json::CharReaderBuilder readBuilder;
        std::ifstream ifs(configFile);
        std::unique_ptr<Json::CharReader> reader(readBuilder.newCharReader());
        JSONCPP_STRING errs;
        if (Json::parseFromStream(readBuilder, ifs, &rootReder, &errs)) {
            if (rootReder.isMember("LogAppName")) {
                LogAppName = rootReder["LogAppName"].asString();
            }
            if (rootReder.isMember("LogAppDescription")) {
                LogAppDescription = rootReder["LogAppDescription"].asString();
            }
            if (rootReder.isMember("LogContextName")) {
                LogContextName = rootReder["LogContextName"].asString();
            }
            if (rootReder.isMember("LogFilePath")) {
                LogFilePath = rootReder["LogFilePath"].asString();
            }
            if (rootReder.isMember("LogLevel")) {
                LogLevel = rootReder["LogLevel"].asUInt();
            }
            if (rootReder.isMember("LogMode")) {
                LogMode = rootReder["LogMode"].asUInt();
            }
            if (rootReder.isMember("MaxLogFileNum")) {
                MaxLogFileNum = rootReder["MaxLogFileNum"].asUInt();
            }
            if (rootReder.isMember("MaxSizeOfLogFile")) {
                MaxSizeOfLogFile = rootReder["MaxSizeOfLogFile"].asUInt();
            }
            if (rootReder.isMember("PerFilePath")) {
                PerFilePath = rootReder["PerFilePath"].asString();
            }
            if (rootReder.isMember("BakPerFilePath")) {
                BakPerFilePath = rootReder["BakPerFilePath"].asString();
            }
            if (rootReder.isMember("MaxComValLimit")) {
                MaxComValLimit = rootReder["MaxComValLimit"].asUInt();
            }
        } else {
            cfgparseflag = false;
        }
    } else {
        cfgfoundflag = false;
    }
#ifdef BUILD_FOR_MDC
#elif BUILD_FOR_ORIN
#else
    LogFilePath = "./";
    PerFilePath = "./cfg/";
    BakPerFilePath = "./cfg/";
#endif
    ConfigLogger::GetInstance().InitLogging(LogAppName,                               // the id of application
                                            LogAppDescription,                        // the log id of application
                                            ConfigLogger::CFGLogLevelType(LogLevel),  // the log level of application
                                            LogMode,                                  // the output log mode
                                            LogFilePath,                              // the log file directory, active when output log to file
                                            MaxLogFileNum,                            // the max number log file , active when output log to file
                                            MaxSizeOfLogFile);
    CONFIG_LOG_INFO << "LogAppName : " << LogAppName.c_str();
    CONFIG_LOG_INFO << "LogAppDescription : " << LogAppDescription.c_str();
    CONFIG_LOG_INFO << "LogContextName : " << LogContextName.c_str();
    CONFIG_LOG_INFO << "LogFilePath : " << LogFilePath.c_str();
    CONFIG_LOG_INFO << "LogLevel : " << LogLevel;
    CONFIG_LOG_INFO << "LogMode : " << LogMode;
    CONFIG_LOG_INFO << "MaxLogFileNum : " << MaxLogFileNum;
    CONFIG_LOG_INFO << "MaxSizeOfLogFile : " << MaxSizeOfLogFile;
    CONFIG_LOG_INFO << "PerFilePath : " << PerFilePath.c_str();
    CONFIG_LOG_INFO << "BakPerFilePath : " << BakPerFilePath.c_str();
    CONFIG_LOG_INFO << "MaxComValLimit : " << MaxComValLimit;
#ifdef BUILD_FOR_ORIN
    ara::core::Initialize();
#endif
    // CfgServer server;
    CfgServerProto server;
    server.Init(PerFilePath, BakPerFilePath, MaxComValLimit);
    std::shared_ptr<ExecClient> execli(new ExecClient());
    int32_t ret = execli->ReportState(ExecutionState::kRunning);
    if (ret) {
        CONFIG_LOG_WARN << "cfg report fail.kRunning";
    } else {
        CONFIG_LOG_INFO << "cfg report succ.kRunning";
    }
    PhmClientInstance::getInstance()->Init();
    if (cfgparseflag == false) {
        CONFIG_LOG_WARN << "configfile parse error " << configFile.c_str();
        SendFault_t fault(4180, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
    }
    if (cfgfoundflag == false) {
        CONFIG_LOG_WARN << "configfile is not found: " << configFile.c_str();
        SendFault_t fault(4170, 1, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
    }
    server.Run();
    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    server.DeInit();
#ifdef BUILD_FOR_ORIN
    ara::core::Deinitialize();
#endif
    ret = execli->ReportState(ExecutionState::kTerminating);
    if (ret) {
        CONFIG_LOG_WARN << "cfg report fail.kTerminating";
    } else {
        CONFIG_LOG_INFO << "cfg report succ.kTerminating";
    }
    PhmClientInstance::getInstance()->DeInit();
    return 0;
}

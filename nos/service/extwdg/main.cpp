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

#include "em/include/exec_client.h"
#include "em/include/proctypes.h"
#include "json/json.h"
#include "extwdg_logger.h"
#include "extwdg.h"

using namespace hozon::netaos::extwdg;
using namespace hozon::netaos::em;

#define EW_SERVER_CONFIG_FILE_DEFAULT ("/app/conf/extwdg_config.json")

sig_atomic_t g_stopFlag = 0;

void SigHandler(int signum) {
    g_stopFlag = 1;
    EW_INFO << "--- extwdg SigHandler enter, signum [" << signum << "] ---";

}

static void InitLog() {
    std::string configFile = EW_SERVER_CONFIG_FILE_DEFAULT;
    std::string LogAppName = "EW";
    std::string LogAppDescription = "extwdg";
    std::string LogContextName = "EW";
    std::string LogFilePath = "/opt/usr/log/soc_log/";
    uint32_t LogLevel = 1;  // kDebug
    uint32_t LogMode = 2;   // Console && File
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

    EWServerLogger::GetInstance().InitLogging(LogAppName,                                    // the id of application
                                                LogAppDescription,                             // the log id of application
                                                EWServerLogger::EWLogLevelType(LogLevel),  //the log level of application
                                                LogMode,                                       //the output log mode
                                                LogFilePath,                                   //the log file directory, active when output log to file
                                                MaxLogFileNum,                                 //the max number log file , active when output log to file
                                                MaxSizeOfLogFile                               //the max size of each  log file , active when output log to file
    );
    EWServerLogger::GetInstance().CreateLogger(LogContextName);
}

static std::shared_ptr<ExtWdg>
CreatExtwdg()
{
    std::string configFile = EW_SERVER_CONFIG_FILE_DEFAULT;
    Json::Value TransInfoArray;
    TransportInfo info;
    std::vector<TransportInfo> transinfos;
    uint32_t TransInfonumb = 1;
    if (0 == access(configFile.c_str(), F_OK)) {
        Json::Value jsonData;
        Json::CharReaderBuilder readBuilder;
        std::ifstream ifs(configFile);
        std::unique_ptr<Json::CharReader> reader(readBuilder.newCharReader());
        JSONCPP_STRING errs;
        if (Json::parseFromStream(readBuilder, ifs, &jsonData, &errs)) {
            TransInfonumb = (jsonData["TransInfonumb"]) ? jsonData["TransInfonumb"].asUInt() : TransInfonumb;
            if(jsonData.isMember("TransInfo")&& jsonData["TransInfo"].isArray()) {
                TransInfoArray = jsonData["TransInfo"];
            }
            else {
                EW_ERROR << "ReadTransInfo FAILED! and no transinfo config here";
            }
            EW_INFO << "TransInfoArray.size() is " <<TransInfoArray.size();
            for (uint32_t i = 0; i < TransInfonumb; ++i) {
                Json::Value transInfo = TransInfoArray[i];
                info.dest_ip = (transInfo["DestIp"]) ? transInfo["DestIp"].asString() : info.dest_ip;
                info.dest_port = (transInfo["DestPort"]) ? transInfo["DestPort"].asInt() : info.dest_port;
                info.host_ip = (transInfo["HostIP"]) ? transInfo["HostIP"].asString() : info.host_ip;
                info.host_port = (transInfo["HostPort"]) ? transInfo["HostPort"].asInt() : info.host_port;
                info.protocol = (transInfo["Protocol"]) ? transInfo["Protocol"].asString() : info.protocol;

            }

            transinfos.emplace_back(info);

            for (uint32_t i = 0; i < TransInfonumb; ++i)
            {
                EW_INFO << "transinfos data: dest ip is " << transinfos[i].dest_ip 
                                        << "dest port is " << transinfos[i].dest_port
                                        << "host ip is " << transinfos[i].host_ip
                                        << "host port is " <<  transinfos[i].host_port
                                        << "protocol is " << transinfos[i].protocol;

            }

            std::shared_ptr<ExtWdg> extwdg = std::make_shared<ExtWdg>(transinfos);
            return extwdg;
        }
    }
    
}

int main(int argc, char** argv) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);
    InitLog();
    std::shared_ptr<ExtWdg> extwdg = CreatExtwdg();

    if(extwdg->Init() != 0)
    {
        EW_ERROR << "ExtWdg::Init() FAILED!";
    }
    extwdg->Run();

    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    EW_INFO << "g_stopFlag comming!";
    extwdg->Stop();
    extwdg->DeInit();
    EW_INFO << "main before return!";
    return 0;
}

/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     hz_log_manager.cpp                                                     *
*  @brief    Implement of class HzTrace and  HzLogManager   *
*  Details.                                                                         *
*                                                                                   *
*  @version  0.0.0.1                                                                *
*                                                                                   *
*-----------------------------------------------------------------------------------*
*  Change History :                                                                 *
*  <Date>     | <Version> | <Author>       | <Description>                          *
*-----------------------------------------------------------------------------------*
*  2023/12/06 | 0.0.0.1   | XuMengJun      | Create file                            *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#include <cstring>
#include <fstream>

#include "json/json.h"

#include "global_log_config_manager.h"

namespace hozon {
namespace netaos {
namespace log {

HzGlobalLogConfigManager::HzGlobalLogConfigManager() {
}

HzGlobalLogConfigManager::~HzGlobalLogConfigManager() {
}

HzGlobalLogConfigManager& HzGlobalLogConfigManager::GetInstance() {
    static HzGlobalLogConfigManager log_config;
    return log_config;
}

bool HzGlobalLogConfigManager::LoadConfig() {
    if (hasLoadFile) {
        return true;
    }

    std::ifstream infile(global_log_config_file, std::ios::binary);
    if (!infile.is_open()) {
        return false;
    }

    Json::CharReaderBuilder read_builder;
    Json::Value root;
    std::string err;

    if (!Json::parseFromStream(read_builder, infile, &root, &err)) {
        infile.close();
        return false;
    }

    if (root["log_config"].isNull()) {
        infile.close();
        return false;
    }

    int config_size = root["log_config"].size();
    auto &config_value = root["log_config"];
    for (int i = 0; i < config_size; ++i) {
        if (!config_value[i].isMember("LogAppName")) {
            continue;
        }
        LogConfigPtr log_config = std::make_shared<LogConfig>();

        log_config->appId = config_value[i]["LogAppName"].asString();
        log_config->hasAppId = true;

        if (config_value[i].isMember("LogFilePath")) {
            log_config->logPath = config_value[i]["LogFilePath"].asString();
            if (log_config->logPath.back() != '/') {
                log_config->logPath += "/";
            }
            log_config->hasLogPath = true;
        }

        if (config_value[i].isMember("MaxLogFileNum")) {
            log_config->maxLogFileNum = config_value[i]["MaxLogFileNum"].asInt();
            log_config->hasMaxLogFileNum = true;
        }
        if (config_value[i].isMember("MaxSizeOfLogFile")) {
            log_config->maxSizeOfEachLogFile = config_value[i]["MaxSizeOfLogFile"].asInt() * 1024 * 1024;
            log_config->hasMaxSizeOfEachLogFile = true;
        }
        if (config_value[i].isMember("LogMode")) {
            log_config->logMode = config_value[i]["LogMode"].asInt();
            log_config->hasLogMode = true;
        }
        if (config_value[i].isMember("AppLogLevel")) {
            int appLogLevel = config_value[i]["AppLogLevel"].asInt();
            if (appLogLevel < (int)LogLevel::kTrace) {
                appLogLevel = (int)LogLevel::kTrace;
            } else if (appLogLevel > (int)LogLevel::kOff) {
                appLogLevel = (int)LogLevel::kOff;
            }
            log_config->appLogLevel = (hozon::netaos::log::LogLevel)appLogLevel;
            log_config->hasLogLevel = true;
        }
        if (config_value[i].isMember("CtxLogLevels")) {
            const auto &ctxLogLevels = config_value[i]["CtxLogLevels"];
            const auto &memberNames = ctxLogLevels.getMemberNames();
            for (const auto &memberName : memberNames) {
                int ctxLogLevel = ctxLogLevels[memberName].asInt();
                if (ctxLogLevel < (int)LogLevel::kTrace) {
                    ctxLogLevel = (int)LogLevel::kTrace;
                } else if (ctxLogLevel > (int)LogLevel::kOff) {
                    ctxLogLevel = (int)LogLevel::kOff;
                }

                log_config->ctxIdLogLevelMap_.insert(std::make_pair(memberName, (LogLevel)ctxLogLevel));
            }
        }
        appIdLogConfigMap_.insert(std::make_pair(log_config->appId, log_config));
    }

    infile.close();

    hasLoadFile = true;
    return true;
}

const HzGlobalLogConfigManager::LogConfigPtr& HzGlobalLogConfigManager::GetAppLogConfig(const std::string &appId) {
    return appIdLogConfigMap_[appId];
}

} // namespace log
} // namespace netaos
} // namespace hozon

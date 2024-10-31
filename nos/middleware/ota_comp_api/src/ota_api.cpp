/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: ota api definition
 */
#include <iostream>
#include <fstream>
#include <mutex>
#include <unistd.h>

#include "json/json.h"
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "ota_api_logger.h"
#include "ota_api.h"
#include "ota_api_zmq.h"

using namespace hozon::netaos::em;
using namespace hozon::netaos::otaapi;
std::shared_ptr<ExecClient> spExecClient = nullptr;
OTAApiZmq *otaapi = nullptr;
std::mutex mutex_otaapi;


#define OTA_API_CONFIG_FILE_DEFAULT        ("/app/conf/ota_api_config.json")
static void InitLog()
{
    std::string configFile = OTA_API_CONFIG_FILE_DEFAULT;
    std::string LogAppName = "OTA_API";
    std::string LogAppDescription = "ota_api";
    std::string LogContextName = "OTA_API";
    std::string LogFilePath = "/opt/usr/log/soc_log/";
    uint32_t    LogLevel = 1;  // kDebug
    uint32_t    LogMode = 2;   // Console && File
    uint32_t    MaxLogFileNum = 10;
    uint32_t    MaxSizeOfLogFile = 5;

    if (0 == access(configFile.c_str(), F_OK)) {
        Json::Value rootReder;
        Json::CharReaderBuilder readBuilder;
        std::ifstream ifs(configFile);
        std::unique_ptr<Json::CharReader> reader(readBuilder.newCharReader());
        JSONCPP_STRING errs;
        if (Json::parseFromStream(readBuilder, ifs, &rootReder, &errs)) {
            LogAppName = ("" != rootReder["LogAppName"].asString()) ? rootReder["LogAppName"].asString() : LogAppName;
            LogAppDescription = ("" != rootReder["LogAppDescription"].asString()) ? rootReder["LogAppDescription"].asString() : LogAppDescription;
            LogContextName = ("" != rootReder["LogContextName"].asString()) ? rootReder["LogContextName"].asString() : LogContextName;
            LogFilePath = ("" != rootReder["LogFilePath"].asString()) ? rootReder["LogFilePath"].asString() : LogFilePath;
            LogLevel = (0 != rootReder["LogLevel"].asUInt()) ? rootReder["LogLevel"].asUInt() : LogLevel;
            LogMode = (0 != rootReder["LogMode"].asUInt()) ? rootReder["LogMode"].asUInt() : LogMode;
            MaxLogFileNum = (0 != rootReder["MaxLogFileNum"].asUInt()) ? rootReder["MaxLogFileNum"].asUInt() : MaxLogFileNum;
            MaxSizeOfLogFile = (0 != rootReder["MaxSizeOfLogFile"].asUInt()) ? rootReder["MaxSizeOfLogFile"].asUInt() : MaxSizeOfLogFile;
        }
    }

    OtaApiLogger::GetInstance().InitLogging(LogAppName,    // the id of application
        LogAppDescription, // the log id of application
        OtaApiLogger::OTAAPILogLevelType(LogLevel), // the log level of application
        LogMode, // the output log mode
        LogFilePath, // the log file directory, active when output log to file
        MaxLogFileNum, // the max number log file , active when output log to file
        MaxSizeOfLogFile // the max size of each  log file , active when output log to file
    );
    OtaApiLogger::GetInstance().CreateLogger(LogContextName);
}

enum STD_RTYPE_E ota_api_init()
{
    std::lock_guard<std::mutex> lock(mutex_otaapi);
    InitLog();
    OTA_API_LOG_INFO << "ota_api_init";
    if (spExecClient == nullptr) {
        spExecClient = std::make_shared<ExecClient>();
        spExecClient->ReportState(ExecutionState::kRunning);
    }
    if (otaapi == nullptr) {
        otaapi = new OTAApiZmq();
        otaapi->ota_api_init();
    }

    return E_OK;
}

enum STD_RTYPE_E ota_api_deinit()
{
    std::lock_guard<std::mutex> lock(mutex_otaapi);
    OTA_API_LOG_INFO << "ota_api_deinit";
    if (otaapi != nullptr) {
        otaapi->ota_api_deinit();
        delete otaapi;
        otaapi = nullptr;
    }
    if (spExecClient != nullptr) {
        spExecClient->ReportState(ExecutionState::kTerminating);
        spExecClient = nullptr;
    }

    return E_OK;
}

enum STD_RTYPE_E ota_get_api_version(int8_t *api_version)
{
    OTA_API_LOG_INFO << "ota_get_api_version";
    if (api_version == nullptr) {
        OTA_API_LOG_WARN << "otaapi or api_version is nullptr";
        return E_NOT_OK;
    }
    memcpy(api_version, (int8_t *)API_VERSION, strlen(API_VERSION));
    
    return E_OK;
}

enum STD_RTYPE_E ota_start_update(uint8_t *file_path)
{
    OTA_API_LOG_INFO << "ota_start_update";
    if (otaapi == nullptr || file_path == nullptr) {
        OTA_API_LOG_WARN << "otaapi or file_path is nullptr";
        return E_NOT_OK;
    }

    otaapi->ota_get_version();

    if (otaapi->ota_precheck() < 0) {
        return E_NOT_OK;
    }

    std::string str_file((const char *)file_path);
    if (otaapi->ota_start_update(str_file) < 0) {
        return E_NOT_OK;
    }

    return E_OK;
}
enum STD_RTYPE_E ota_get_update_status(enum OTA_UPDATE_STATUS_E *ota_update_status, uint8_t *progress)
{
    OTA_API_LOG_INFO << "ota_get_update_status";
    if (otaapi == nullptr || ota_update_status == nullptr || progress == nullptr) {
        OTA_API_LOG_WARN << "otaapi or file_path is nullptr";
        return E_NOT_OK;
    }

    int32_t ota_status = otaapi->ota_get_update_status();
    uint8_t ota_progress = otaapi->ota_progress();
    if (ota_status < 0) {
        return E_NOT_OK;
    }

    if (ota_status == State::NORMAL_IDLE) {
        *ota_update_status = OTA_UPDATE_STATUS_IDLE;
    }
    else if (ota_status >= State::OTA_PRE_UPDATE && ota_status <= State::OTA_ACTIVING) {
        *ota_update_status = OTA_UPDATE_STATUS_PROCESSING;
    }
    else if (ota_status == State::OTA_ACTIVED) {
        *ota_update_status = OTA_UPDATE_STATUS_SUCCESS;
    }
    else if (ota_status == State::OTA_UPDATE_FAILED) {
        *ota_update_status = OTA_UPDATE_STATUS_FAIL;
    }
    *progress = ota_progress;

    return E_OK;
}
enum STD_RTYPE_E ota_get_log_path (uint8_t *log_path)
{
    return E_OK;
}
enum STD_RTYPE_E ota_log_callback_register (OTA_LOG_HANDLER ota_log_handler)
{
    return E_OK;
}




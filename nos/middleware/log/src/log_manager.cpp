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
*  2022/06/15 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#include <iostream>
#include <fstream>
#include <memory>

#include "log_manager.hpp"
#include "adapter/hz_log_trace.hpp"
#include "hz_logger.hpp"
#include "log/include/global_log_config_manager.h"

namespace hozon {
namespace netaos {
namespace log {

HzLogManager* HzLogManager::instance_ = nullptr;
std::mutex HzLogManager::mtx_;


HzLogManager* HzLogManager::GetInstance() {
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new HzLogManager();
        }
    }
    return instance_;
}

std::shared_ptr<Logger> HzLogManager::creatLogger(std::string ctxId, std::string ctxDescription,
                                                  LogLevel ctxDefLogLevel, uint32_t& loggerCount)
{
    std::lock_guard<std::mutex> lck(mtx_);
    auto it = Logger_.begin();
    for(; it != Logger_.end(); it++) {
        if (ctxId == it->get()->GetCtxId()) {
            loggerCount = Logger_.size();
            break;
        }
    }

    if (it != Logger_.end()) {
        // Already creted a same module logger, get it.
        return std::static_pointer_cast<Logger>(*it);
    }
    else {
        // default ctx output level should >= app log level
        LogLevel ctxLogLevel = (ctxDefLogLevel >= appLogLevel_) ? ctxDefLogLevel : appLogLevel_;

        // global config set ctx log level
        if (HzGlobalLogConfigManager::GetInstance().LoadConfig()) {
            const auto &log_config = HzGlobalLogConfigManager::GetInstance().GetAppLogConfig(appId_);
            if (log_config && appId_ == ctxId) {
                ctxLogLevel = log_config->appLogLevel;
            }

            if (log_config && log_config->ctxIdLogLevelMap_.count(ctxId) > 0) {
                ctxLogLevel = log_config->ctxIdLogLevelMap_[ctxId];
            }
        }

        std::string levelInfo[7] = { "trace", "debug", "info", "warn", "error", "critical", "off" };
        if (NULL != getenv(std::string(ctxId + "_LOG_LEVEL" ).c_str())) {
            std::string levelstr = std::string(getenv(std::string(ctxId + "_LOG_LEVEL" ).c_str()));
            if (std::string("") != levelstr && (std::string(" ") != levelstr)) {
                ctxLogLevel = static_cast<LogLevel>(strtol(levelstr.c_str(), NULL, 0));
                for (int i = 0; i < 7; ++i) {
                    if (std::string(levelInfo[i]) == levelstr ) {
                        ctxLogLevel = static_cast<LogLevel>(i);
                        break;
                    }
                }
            }
        }
        // Haven't creted a same module logger, create a new one.
        std::shared_ptr<HzLogger> new_logger = std::make_shared<HzLogger>(ctxId, ctxDescription, ctxLogLevel);

        // Store the new logger to list.
        Logger_.push_back(new_logger);
        loggerCount++;

        return std::static_pointer_cast<Logger>(new_logger);
    }
}

std::shared_ptr<Logger> HzLogManager::CreateOperationLogger(std::string ctxId, std::string ctxDescription,
                                                  LogLevel ctxDefLogLevel, uint32_t& loggerCount)
{
    std::lock_guard<std::mutex> lck(mtx_);
    auto it = OperationLogger_.begin();
    for(; it != OperationLogger_.end(); it++) {
        if (ctxId == it->get()->GetCtxId()) {
            loggerCount = OperationLogger_.size();
            break;
        }
    }

    if (it != OperationLogger_.end()) {
        // Already creted a same module logger, get it.
        return std::static_pointer_cast<Logger>(*it);
    }
    else {
        LogLevel ctxLogLevel = ctxDefLogLevel;
        if (HzGlobalLogConfigManager::GetInstance().LoadConfig()) {
            const auto &log_config = HzGlobalLogConfigManager::GetInstance().GetAppLogConfig(appId_);
            if (log_config && log_config->ctxIdLogLevelMap_.count(ctxId) > 0) {
                ctxLogLevel = log_config->ctxIdLogLevelMap_[ctxId];
            }
        }

        std::string levelInfo[7] = { "trace", "debug", "info", "warn", "error", "critical", "off" };
        if (NULL != getenv(std::string(ctxId + "_LOG_LEVEL" ).c_str())) {
            std::string levelstr = std::string(getenv(std::string(ctxId + "_LOG_LEVEL" ).c_str()));
            if (std::string("") != levelstr && (std::string(" ") != levelstr)) {
                ctxLogLevel = static_cast<LogLevel>(strtol(levelstr.c_str(), NULL, 0));
                for (int i = 0; i < 7; ++i) {
                    if (std::string(levelInfo[i]) == levelstr ) {
                        ctxLogLevel = static_cast<LogLevel>(i);
                        break;
                    }
                }
            }
        }

        //Haven't creted a same module logger, create a new one.
        std::shared_ptr<HzOperationLogger> new_logger = std::make_shared<HzOperationLogger>(ctxId, ctxDescription, ctxLogLevel);

        // Store the new logger to list.
        OperationLogger_.push_back(new_logger);
        loggerCount++;

        return std::static_pointer_cast<Logger>(new_logger);
    }
}

void HzLogManager::InitLogging(std::string appId, std::string appDescription, LogLevel appDefLogLevel,
                               std::uint32_t mode, std::string directoryPath, std::uint32_t maxLogFileNum,
                               std::uint64_t maxSizeOfLogFile, bool pureLogFormat)
{
    std::lock_guard<std::mutex> lck(mtx_);

    if (maxSizeOfLogFile > (20 * 1024 * 1024)) {
        maxSizeOfLogFile = (20 * 1024 * 1024);
    }

    if (maxSizeOfLogFile < (5 * 1024 * 1024)) {
        maxSizeOfLogFile = (5 * 1024 * 1024);
    }

    appId_ = appId;
    appDescription_ = appDescription;
    appLogLevel_ = appDefLogLevel;
    mode_ = mode;
    directoryPath_ = fixPath(directoryPath);
    logFileBaseName_ = appId_; // Log file base name
    maxLogFileStoredNum_ = maxLogFileNum ; // max log file stored number
    maxSizeOfEachLogFile_ = maxSizeOfLogFile;  // max size of each log file
    pureLogFormat_ = pureLogFormat;

    if (HzGlobalLogConfigManager::GetInstance().LoadConfig()) {
        const auto &log_config = HzGlobalLogConfigManager::GetInstance().GetAppLogConfig(appId_);
        if (log_config && log_config->hasLogLevel) {
            appLogLevel_ = log_config->appLogLevel;
        }
        if (log_config && log_config->hasLogMode) {
            mode_ = log_config->logMode;
        }
        if (log_config && log_config->hasLogPath) {
            directoryPath_ = log_config->logPath;
        }
        if (log_config && log_config->hasMaxLogFileNum) {
            maxLogFileStoredNum_ = log_config->maxLogFileNum;
        }
        if (log_config && log_config->hasMaxSizeOfEachLogFile) {
            maxSizeOfEachLogFile_ = log_config->maxSizeOfEachLogFile;
        }
    }

    std::string levelInfo[7] = { "trace", "debug", "info", "warn", "error", "critical", "off" };
    if (NULL != getenv(std::string(appId + "_LOG_LEVEL" ).c_str())) {
        std::string levelstr = std::string(getenv(std::string(appId + "_LOG_LEVEL" ).c_str()));
        if (std::string("") != levelstr && (std::string(" ") != levelstr)) {
            appLogLevel_ = static_cast<LogLevel>(strtol(levelstr.c_str(), NULL, 0));
            for (int i = 0; i < 7; ++i) {
                if (std::string(levelInfo[i]) == levelstr ) {
                    appLogLevel_ = static_cast<LogLevel>(i);
                    break;
                }
            }
        }
    }

    /* Should update the log output level of already created log. */
    // auto it = Logger_.begin();
    // for(; it != Logger_.end(); it++) {
    //     it->get()->UpdateAppLogLevel(appLogLevel_);
    // }

    // auto it_opt = OperationLogger_.begin();
    // for(; it_opt != OperationLogger_.end(); it_opt++) {
    //     it_opt->get()->UpdateAppLogLevel(appLogLevel_);
    // }

    // Create hozon trace
    if (nullptr == LoggerTrace_) {
        LoggerTrace_ = std::make_shared<HzLogTrace>();
        LoggerTrace_->setFileName("hz_log_file");
        LoggerTrace_->setTerminalName("hz_log_tml");
    }

    if ((mode_ & HZ_LOG2CONSOLE) > 0) {
        LoggerTrace_->setLog2Terminal(true);
    }
    else{
        LoggerTrace_->setLog2Terminal(false);
        if (NULL != getenv(std::string(appId + "_LOG_CONSOLE" ).c_str())
            && std::string("") != std::string(getenv(std::string(appId + "_LOG_CONSOLE" ).c_str())) ) {
            if (std::string("1") == std::string(getenv(std::string(appId + "_LOG_CONSOLE" ).c_str()))
                || std::string("true") == std::string(getenv(std::string(appId + "_LOG_CONSOLE" ).c_str())) ) {
                    LoggerTrace_->setLog2Terminal(true);
            }
        }
    }

    if ((mode_ & HZ_LOG2FILE) > 0) {
        LoggerTrace_->setLog2File(true, directoryPath_, logFileBaseName_, maxLogFileStoredNum_, maxSizeOfEachLogFile_);
    }
    else{
        LoggerTrace_->setLog2File(false, directoryPath_, logFileBaseName_, maxLogFileStoredNum_, maxSizeOfEachLogFile_);
    }

    if ((mode_ & HZ_LOG2LOGSERVICE) > 0) {
        LoggerTrace_->setLog2LogService(true, logFileBaseName_);
    } else {
        LoggerTrace_->setLog2LogService(false, logFileBaseName_);
    }

    LoggerTrace_->initDevice(pureLogFormat_);
}

void HzLogManager::logout(LogLevel level, const std::string& message)
{
    if (!LoggerTrace_) {
        return;
    }

    if (level == LogLevel::kTrace) {
        LoggerTrace_->trace(message);
    }
    else if (level == LogLevel::kDebug) {
        LoggerTrace_->debug(message);
    }
    else if (level == LogLevel::kInfo) {
        LoggerTrace_->info(message);
    }
    else if (level == LogLevel::kWarn) {
        LoggerTrace_->warn(message);
    }
    else if (level == LogLevel::kError) {
        LoggerTrace_->error(message);
    }
    else if (level == LogLevel::kCritical) {
        LoggerTrace_->critical(message);
    }
    else {
        // do nothing
    }
}

void HzLogManager::setAllCtxLogLevel(LogLevel level)
{
    auto it = Logger_.begin();
    for (; it != Logger_.end(); it++) {
        // std::cout << "HzLogManager::setAllCtxLogLevel(LogLevel level)" << std::endl;
        // std::cout << "ctx:" << it->get()->GetCtxId()  << std::endl;
        it->get()->ForceSetCtxLogLevel(level);
    }

    auto it_opt = OperationLogger_.begin();
    for (; it_opt != OperationLogger_.end(); it_opt++) {
        // std::cout << "HzLogManager::setAllCtxLogLevel(LogLevel level)" << std::endl;
        // std::cout << "ctx:" << it->get()->GetCtxId()  << std::endl;
        it_opt->get()->ForceSetCtxLogLevel(level);
    }

    auto it_mcu = mcuLogger_.begin();
    for (; it_mcu != mcuLogger_.end(); it_mcu++) {
        it_mcu->get()->ForceSetCtxLogLevel(level);
    }
}

void HzLogManager::setAppLogLevel(std::string appId, LogLevel level)
{
    if (appId != appId_) {
        // go on
    } else {
        auto it = Logger_.begin();
        for (; it != Logger_.end(); it++) {
            it->get()->ForceSetCtxLogLevel(level);
        }

        auto it_opt = OperationLogger_.begin();
        for (; it_opt != OperationLogger_.end(); it_opt++) {
            it_opt->get()->ForceSetCtxLogLevel(level);
        }
    }

    if (appId != "DESAY_MCU" && appId != "HZ_MCU") {
        return;
    } else {
        auto it_mcu_app = mcuAppId_.begin();
        for(; it_mcu_app != mcuAppId_.end(); it_mcu_app++) {
            if (*it_mcu_app == appId) {
                break;
            }
        }
        if (it_mcu_app != mcuAppId_.end()) {
            auto it_mcu = mcuLogger_.begin();
            for (; it_mcu != mcuLogger_.end(); it_mcu++) {
                if (appId == it_mcu->get()->GetAppId()) {
                    it_mcu->get()->ForceSetCtxLogLevel(level);
                }
            }
        }
    }
}

void HzLogManager::setCtxLogLevel(std::string ctxId, LogLevel level)
{
    auto it = Logger_.begin();
    for (; it != Logger_.end(); it++) {
        if (ctxId == it->get()->GetCtxId()) {
            it->get()->ForceSetCtxLogLevel(level);
            break;
        }
    }

    auto it_opt = OperationLogger_.begin();
    for (; it_opt != OperationLogger_.end(); it_opt++) {
        if (ctxId == it_opt->get()->GetCtxId()) {
            it_opt->get()->ForceSetCtxLogLevel(level);
            break;
        }
    }

    auto it_mcu = mcuLogger_.begin();
    for (; it_mcu != mcuLogger_.end(); it_mcu++) {
        if (ctxId == it_mcu->get()->GetCtxId()) {
            it_mcu->get()->ForceSetCtxLogLevel(level);
            break;
        }
    }
}

void HzLogManager::setSpecifiedLogLevel(std::string appId, std::string ctxId, LogLevel level)
{
    if (appId != appId_) {
        // go on
    } else {
        auto it = Logger_.begin();
        for (; it != Logger_.end(); it++) {
            if (ctxId == it->get()->GetCtxId()) {
                // std::cout << "HzLogManager::setSpecifiedLogLevel: " << ", Found ctx. " << std::endl;
                it->get()->ForceSetCtxLogLevel(level);
                break;
            }
        }

        auto it_opt = OperationLogger_.begin();
        for (; it_opt != OperationLogger_.end(); it_opt++) {
            if (ctxId == it_opt->get()->GetCtxId()) {
                // std::cout << "HzLogManager::setSpecifiedLogLevel: " << ", Found ctx. " << std::endl;
                it_opt->get()->ForceSetCtxLogLevel(level);
                break;
            }
        }
    }

    if (appId != "DESAY_MCU" && appId != "HZ_MCU") {
        return;
    } else {
        auto it_mcu_app = mcuAppId_.begin();
        for(; it_mcu_app != mcuAppId_.end(); it_mcu_app++) {
            if (*it_mcu_app == appId) {
                break;
            }
        }
        if (it_mcu_app != mcuAppId_.end()) {
            auto it_mcu = mcuLogger_.begin();
            for (; it_mcu != mcuLogger_.end(); it_mcu++) {
                if (ctxId == it_mcu->get()->GetCtxId() && appId == it_mcu->get()->GetAppId()) {
                    it_mcu->get()->ForceSetCtxLogLevel(level);
                    break;
                }
            }
        }
    }
}

std::string HzLogManager::fixPath(const std::string& path)
{
    std::string newPath{path};
    if (!newPath.empty() && newPath.back() != '/') {
        return newPath + '/';
    }
    return newPath;
}

void HzLogManager::initMcuLogging(const std::string& appId)
{
    std::string mcuAppId = appId;
    std::string mcuDirectoryPath{};
    if (mcuAppId == "DESAY_MCU")
    {
        mcuDirectoryPath = "/svp_log/";
    } else if (mcuAppId == "HZ_MCU") {
        mcuDirectoryPath = "/opt/usr/log/mcu_log/";
    }
    std::string mcuLogFileBaseName = mcuAppId;
    std::uint32_t mcuMaxLogFileStoredNum = 10;
    std::uint64_t mcuMaxSizeOfEachLogFile = 10 * 1024 * 1024;
    bool mcuPureLogFormat = false;
    if (mcuLogTraceMap_.count(mcuAppId) > 0) {
        // do nothing
    } else {
        mcuAppId_.emplace_back(mcuAppId);
        auto mcuLoggerTrace = std::make_shared<HzLogTrace>();
        // Name需要和appId关联，保证不一致
        std::string logFile = mcuAppId + "_log_file";
        std::string logTml = mcuAppId + "_log_tml";
        mcuLoggerTrace->setFileName(logFile);
        mcuLoggerTrace->setTerminalName(logTml);

        mcuLoggerTrace->setLog2Terminal(false);

        std::uint32_t log_mode = 2;
        if (HzGlobalLogConfigManager::GetInstance().LoadConfig()) {
            const auto &log_config = HzGlobalLogConfigManager::GetInstance().GetAppLogConfig(mcuAppId);
            if (log_config && log_config->hasLogMode) {
                log_mode = log_config->logMode;
            }
            if (log_config && log_config->hasLogPath) {
                mcuDirectoryPath = log_config->logPath;
            }
            if (log_config && log_config->hasMaxLogFileNum) {
                mcuMaxLogFileStoredNum = log_config->maxLogFileNum;
            }
            if (log_config && log_config->hasMaxSizeOfEachLogFile) {
                mcuMaxSizeOfEachLogFile = log_config->maxSizeOfEachLogFile;
            }
        }

        bool shell_control = false;
        if (NULL != getenv(std::string(mcuAppId + "_LOG_CONSOLE" ).c_str())
            && std::string("") != std::string(getenv(std::string(mcuAppId + "_LOG_CONSOLE" ).c_str())) ) {
            if (std::string("1") == std::string(getenv(std::string(mcuAppId + "_LOG_CONSOLE" ).c_str()))
                || std::string("true") == std::string(getenv(std::string(mcuAppId + "_LOG_CONSOLE" ).c_str())) ) {
                    mcuLoggerTrace->setLog2Terminal(true);
            }
        }

        if (!shell_control) {
            if ((log_mode & HZ_LOG2CONSOLE) > 0) {
                mcuLoggerTrace->setLog2Terminal(true);
            }
        }

        if ((log_mode & HZ_LOG2FILE) > 0) {
            mcuLoggerTrace->setLog2File(true, mcuDirectoryPath, mcuLogFileBaseName, mcuMaxLogFileStoredNum, mcuMaxSizeOfEachLogFile);
        } else {
            mcuLoggerTrace->setLog2File(false, mcuDirectoryPath, mcuLogFileBaseName, mcuMaxLogFileStoredNum, mcuMaxSizeOfEachLogFile);
        }

        if ((log_mode & HZ_LOG2LOGSERVICE) > 0) {
            mcuLoggerTrace->setLog2LogService(true, mcuLogFileBaseName);
        } else {
            mcuLoggerTrace->setLog2LogService(false, mcuLogFileBaseName);
        }

        mcuLoggerTrace->initDevice(mcuPureLogFormat);
        mcuLogTraceMap_.insert(std::make_pair(mcuAppId, mcuLoggerTrace));
    }
}

std::shared_ptr<Logger> HzLogManager::createMcuLogger(const std::string& appId, const std::string& ctxId, const LogLevel& level)
{
    auto it = mcuLogger_.begin();
    for(; it != mcuLogger_.end(); it++) {
        // 需要同时兼顾 appId 和 ctxId，这个是cover两个不同APP有相同CTX的场景
        if (ctxId == it->get()->GetCtxId() && appId == it->get()->GetAppId()) {
            break;
        }
    }

    if (it != mcuLogger_.end()) {
        return std::static_pointer_cast<Logger>(*it);
    }
    else {
        LogLevel ctxLogLevel = level;
        if (HzGlobalLogConfigManager::GetInstance().LoadConfig()) {
            const auto &log_config = HzGlobalLogConfigManager::GetInstance().GetAppLogConfig(appId);
            if (log_config && log_config->ctxIdLogLevelMap_.count(ctxId) > 0) {
                ctxLogLevel = log_config->ctxIdLogLevelMap_[ctxId];
            }
        }

        std::shared_ptr<HzMcuLogger> new_logger = std::make_shared<HzMcuLogger>(appId, ctxId, ctxLogLevel);
        mcuLogger_.push_back(new_logger);
        return std::static_pointer_cast<Logger>(new_logger);
    }
}

void HzLogManager::mcuLogout(const std::string& appId, const LogLevel& level, const std::string& message)
{
    auto it = mcuLogTraceMap_.find(appId);

    if (it != mcuLogTraceMap_.end()) {
        if (level == LogLevel::kTrace) {
            it->second->trace(message);
        }
        else if (level == LogLevel::kDebug) {
            it->second->debug(message);
        }
        else if (level == LogLevel::kInfo) {
            it->second->info(message);
        }
        else if (level == LogLevel::kWarn) {
            it->second->warn(message);
        }
        else if (level == LogLevel::kError) {
            it->second->error(message);
        }
        else if (level == LogLevel::kCritical) {
            it->second->critical(message);
        }
        else {
            // do nothing
        }
    } else {
        // error
    }
}


}
}
}

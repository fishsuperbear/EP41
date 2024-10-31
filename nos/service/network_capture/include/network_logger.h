/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: network_capture
 * Description: log
 * Created on: Oct 7, 2023
 *
 */

#ifndef SERVICE_NETWORK_CAPTURE_INCLUDE_NETWORK_LOGGER_H_
#define SERVICE_NETWORK_CAPTURE_INCLUDE_NETWORK_LOGGER_H_
#pragma once
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace network_capture {
/* network_capture log class */
class NetworkLogger {
 public:
    static NetworkLogger& GetInstance() {
        static NetworkLogger instance;
        return instance;
    }
    ~NetworkLogger() {}
    enum class NETLogLevelType { NETWORK_VERBOSE = 0, NETWORK_DEBUG = 1, NETWORK_INFO = 2, NETWORK_WARN = 3, NETWORK_ERROR = 4, NETWORK_FATAL = 5, NETWORK_OFF = 6 };
    hozon::netaos::log::LogLevel NETParseLogLevel(const int32_t logLevel) {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<NETLogLevelType>(logLevel);
        switch (type) {
            case NETLogLevelType::NETWORK_VERBOSE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case NETLogLevelType::NETWORK_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case NETLogLevelType::NETWORK_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case NETLogLevelType::NETWORK_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case NETLogLevelType::NETWORK_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case NETLogLevelType::NETWORK_FATAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case NETLogLevelType::NETWORK_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "network_capture",                                                // the log id of application
                     std::string appDescription = "network_capture application",                                       // the log id of application
                     NETLogLevelType appLogLevel = NETLogLevelType::NETWORK_INFO,                                      // the log level of application
                     std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
                     std::string directoryPath = "/opt/usr/log/soc_log/",                                                                 // the log file directory, active when output log to file
                     std::uint32_t maxLogFileNum = 10,                                                                 // the max number log file , active when output log to file
                     std::uint64_t maxSizeOfLogFile = 20                                                               // the max size of each  log file , active when output log to file
    ) {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = NETParseLogLevel(static_cast<int32_t>(appLogLevel));
        hozon::netaos::log::InitLogging(appId, appDescription, applevel, outputMode, directoryPath, maxLogFileNum, maxSizeOfLogFile, true);
    }

    // context regist diagserver
    void CreateLogger(const std::string ctxId) {
        const hozon::netaos::log::LogLevel level = NETParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

 private:
    NetworkLogger() {}
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
    int32_t level_ = static_cast<int32_t>(NETLogLevelType::NETWORK_INFO);
};

#define NETWORK_LOG_HEAD        \
    " pid:" << getpid() << " " \
            << "tid:" << (int64_t)syscall(__NR_gettid) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define NETWORK_LOG_CRITICAL hozon::netaos::network_capture::NetworkLogger::GetInstance().GetLogger()->LogCritical() << NETWORK_LOG_HEAD
#define NETWORK_LOG_ERROR hozon::netaos::network_capture::NetworkLogger::GetInstance().GetLogger()->LogError() << NETWORK_LOG_HEAD
#define NETWORK_LOG_WARN hozon::netaos::network_capture::NetworkLogger::GetInstance().GetLogger()->LogWarn() << NETWORK_LOG_HEAD
#define NETWORK_LOG_INFO hozon::netaos::network_capture::NetworkLogger::GetInstance().GetLogger()->LogInfo() << NETWORK_LOG_HEAD
#define NETWORK_LOG_DEBUG hozon::netaos::network_capture::NetworkLogger::GetInstance().GetLogger()->LogDebug() << NETWORK_LOG_HEAD
#define NETWORK_LOG_TRACE hozon::netaos::network_capture::NetworkLogger::GetInstance().GetLogger()->LogTrace() << NETWORK_LOG_HEAD

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_NETWORK_CAPTURE_INCLUDE_NETWORK_LOGGER_H_

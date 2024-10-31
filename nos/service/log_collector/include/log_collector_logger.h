//
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
//
/// @file log_collector_logger.h
/// @brief
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_LOG_COLLECTOR_LOGGER_H__
#define __LOG_COLLECTOR_INCLUDE_LOG_COLLECTOR_LOGGER_H__

#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>

#include "log/include/logging.h"
#include "log_server/common/log_server_def.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class LogCollectorLogger
{
public:
    LogCollectorLogger() : logger_(nullptr) {};
    virtual ~LogCollectorLogger() {};

    enum class LogCollectorLogLevelType {
        LOG_LEVEL_TRACE = 0,
        LOG_LEVEL_DEBUG = 1,
        LOG_LEVEL_INFO = 2,
        LOG_LEVEL_WARN = 3,
        LOG_LEVEL_ERROR = 4,
        LOG_LEVEL_CRITICAL = 5,
        LOG_LEVEL_OFF = 6
    };


    hozon::netaos::log::LogLevel LogCollectorParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<LogCollectorLogLevelType>(logLevel);
        switch (type) {
            case LogCollectorLogLevelType::LOG_LEVEL_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case LogCollectorLogLevelType::LOG_LEVEL_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case LogCollectorLogLevelType::LOG_LEVEL_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case LogCollectorLogLevelType::LOG_LEVEL_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case LogCollectorLogLevelType::LOG_LEVEL_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case LogCollectorLogLevelType::LOG_LEVEL_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case LogCollectorLogLevelType::LOG_LEVEL_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "DEFAULT_APP",  // the log id of application
        std::string appDescription = "default application", // the log id of application
        LogCollectorLogLevelType appLogLevel = LogCollectorLogLevelType::LOG_LEVEL_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = LogCollectorParseLogLevel(static_cast<int32_t> (appLogLevel));
        hozon::netaos::log::InitLogging(
            appId,
            appDescription,
            applevel,
            outputMode,
            directoryPath,
            maxLogFileNum,
            maxSizeOfLogFile,
            true
        );
    }

    void CreateLogger(const std::string ctxId)
    {
        const hozon::netaos::log::LogLevel level = LogCollectorParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static LogCollectorLogger& GetInstance()
    {
        static LogCollectorLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> LogCollectorGetLogger() const
    {
        return logger_;
    }

    int32_t getLogLevel()
    {
        return level_;
    }

    void setLogLevel(int32_t level)
    {
        level_ = level;
    }

private:
    LogCollectorLogger(const LogCollectorLogger&);
    LogCollectorLogger& operator=(const LogCollectorLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(LogCollectorLogLevelType::LOG_LEVEL_INFO);
};

#define LOG_COLLECTOR_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define LOG_COLLECTOR_CRITICAL if(LogCollectorLogger::GetInstance().LogCollectorGetLogger())LogCollectorLogger::GetInstance().LogCollectorGetLogger()->LogCritical() << LOG_COLLECTOR_HEAD << hozon::netaos::log::FROM_HERE
#define LOG_COLLECTOR_ERROR if(LogCollectorLogger::GetInstance().LogCollectorGetLogger())LogCollectorLogger::GetInstance().LogCollectorGetLogger()->LogError() << LOG_COLLECTOR_HEAD << hozon::netaos::log::FROM_HERE
#define LOG_COLLECTOR_WARN if(LogCollectorLogger::GetInstance().LogCollectorGetLogger())LogCollectorLogger::GetInstance().LogCollectorGetLogger()->LogWarn() << LOG_COLLECTOR_HEAD << hozon::netaos::log::FROM_HERE
#define LOG_COLLECTOR_INFO if(LogCollectorLogger::GetInstance().LogCollectorGetLogger())LogCollectorLogger::GetInstance().LogCollectorGetLogger()->LogInfo() << LOG_COLLECTOR_HEAD << hozon::netaos::log::FROM_HERE
#define LOG_COLLECTOR_DEBUG if(LogCollectorLogger::GetInstance().LogCollectorGetLogger())LogCollectorLogger::GetInstance().LogCollectorGetLogger()->LogDebug() << LOG_COLLECTOR_HEAD << hozon::netaos::log::FROM_HERE
#define LOG_COLLECTOR_TRACE if(LogCollectorLogger::GetInstance().LogCollectorGetLogger ())LogCollectorLogger::GetInstance().LogCollectorGetLogger()->LogTrace() << LOG_COLLECTOR_HEAD << hozon::netaos::log::FROM_HERE

}  // namespace logcollector
}  // namespace netaos
}  // namespace hozon

#endif // __LOG_COLLECTOR_INCLUDE_LOG_COLLECTOR_LOGGER_H__

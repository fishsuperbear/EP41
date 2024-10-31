/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: system monitor logger
 */

#ifndef SYSTEM_MONITOR_LOGGER_H_
#define SYSTEM_MONITOR_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"
#include "system_monitor/include/common/to_string.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

class SystemMonitorLogger
{
public:
    SystemMonitorLogger() : logger_(nullptr) {};
    virtual ~SystemMonitorLogger() {};

    enum class SystemMonitorLogLevelType {
        SYSTEM_MONITOR_TRACE = 0,
        SYSTEM_MONITOR_DEBUG = 1,
        SYSTEM_MONITOR_INFO = 2,
        SYSTEM_MONITOR_WARN = 3,
        SYSTEM_MONITOR_ERROR = 4,
        SYSTEM_MONITOR_CRITICAL = 5,
        SYSTEM_MONITOR_OFF = 6
    };


    hozon::netaos::log::LogLevel ParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<SystemMonitorLogLevelType>(logLevel);
        switch (type) {
            case SystemMonitorLogLevelType::SYSTEM_MONITOR_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case SystemMonitorLogLevelType::SYSTEM_MONITOR_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case SystemMonitorLogLevelType::SYSTEM_MONITOR_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case SystemMonitorLogLevelType::SYSTEM_MONITOR_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case SystemMonitorLogLevelType::SYSTEM_MONITOR_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case SystemMonitorLogLevelType::SYSTEM_MONITOR_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case SystemMonitorLogLevelType::SYSTEM_MONITOR_OFF:
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
        SystemMonitorLogLevelType appLogLevel = SystemMonitorLogLevelType::SYSTEM_MONITOR_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20  //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = ParseLogLevel(static_cast<int32_t> (appLogLevel));
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

    // context regist diagserver
    void CreateLogger(const std::string ctxId)
    {
        const hozon::netaos::log::LogLevel level = ParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static SystemMonitorLogger& GetInstance()
    {
        static SystemMonitorLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const
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
    SystemMonitorLogger(const SystemMonitorLogger&);
    SystemMonitorLogger& operator=(const SystemMonitorLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(SystemMonitorLogLevelType::SYSTEM_MONITOR_INFO);
};

#define STMM_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define STMM_CRITICAL (SystemMonitorLogger::GetInstance().GetLogger()->LogCritical() << STMM_HEAD)
#define STMM_ERROR (SystemMonitorLogger::GetInstance().GetLogger()->LogError() << STMM_HEAD)
#define STMM_WARN (SystemMonitorLogger::GetInstance().GetLogger()->LogWarn() << STMM_HEAD)
#define STMM_INFO (SystemMonitorLogger::GetInstance().GetLogger()->LogInfo() << STMM_HEAD)
#define STMM_DEBUG (SystemMonitorLogger::GetInstance().GetLogger()->LogDebug() << STMM_HEAD)
#define STMM_TRACE (SystemMonitorLogger::GetInstance().GetLogger()->LogTrace() << STMM_HEAD)

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon

#endif  // end of SYSTEM_MONITOR_LOGGER_H_
// end of file

#pragma once

#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>

#include "log/include/logging.h"
#include "log_server/common/log_server_def.h"

namespace hozon {
namespace netaos {
namespace logserver {

/* Fm log class */
class LogServerLogger
{
public:
    LogServerLogger() : logger_(nullptr) {};
    virtual ~LogServerLogger() {};

    // only process can use this function
    void InitLogging(std::string appId = "DEFAULT_APP",  // the log id of application
        std::string appDescription = "default application", // the log id of application
        hozon::netaos::log::LogLevel appLevel = hozon::netaos::log::LogLevel::kInfo, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<uint32_t>(appLevel);
        hozon::netaos::log::InitLogging(
            appId,
            appDescription,
            appLevel,
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
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, static_cast<hozon::netaos::log::LogLevel>(level_));
        logger_ = logger;
    }

    static LogServerLogger& GetInstance()
    {
        static LogServerLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> LogServerGetLogger() const
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
    LogServerLogger(const LogServerLogger&);
    LogServerLogger& operator=(const LogServerLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<uint32_t>(hozon::netaos::log::LogLevel::kInfo);
};

#define LOG_SERVER_HEAD getpid() << " " << (long int)syscall(__NR_gettid) << " " <<__FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "
#define LOG_SERVER_CRITICAL LogServerLogger::GetInstance().LogServerGetLogger()->LogCritical() << LOG_SERVER_HEAD
#define LOG_SERVER_ERROR    LogServerLogger::GetInstance().LogServerGetLogger()->LogError() << LOG_SERVER_HEAD
#define LOG_SERVER_WARN     LogServerLogger::GetInstance().LogServerGetLogger()->LogWarn() << LOG_SERVER_HEAD
#define LOG_SERVER_INFO     LogServerLogger::GetInstance().LogServerGetLogger()->LogInfo() << LOG_SERVER_HEAD
#define LOG_SERVER_DEBUG    LogServerLogger::GetInstance().LogServerGetLogger()->LogDebug() << LOG_SERVER_HEAD
#define LOG_SERVER_TRACE    LogServerLogger::GetInstance().LogServerGetLogger()->LogTrace() << LOG_SERVER_HEAD

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

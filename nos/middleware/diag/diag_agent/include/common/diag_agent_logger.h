/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: diag agent logger
 */

#ifndef DIAG_AGENT_LOGGER_H_
#define DIAG_AGENT_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"
#include "diag/diag_agent/include/common/to_string.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace diag_agent {

class DiagAgentLogger
{
public:
    DiagAgentLogger() : logger_(nullptr) {};
    virtual ~DiagAgentLogger() {};

    enum class DiagAngetLogLevelType {
        DIAG_TRACE = 0,
        DIAG_DEBUG = 1,
        DIAG_INFO = 2,
        DIAG_WARN = 3,
        DIAG_ERROR = 4,
        DIAG_CRITICAL = 5,
        DIAG_OFF = 6
    };


    hozon::netaos::log::LogLevel DiagParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<DiagAngetLogLevelType>(logLevel);
        switch (type) {
            case DiagAngetLogLevelType::DIAG_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case DiagAngetLogLevelType::DIAG_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case DiagAngetLogLevelType::DIAG_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case DiagAngetLogLevelType::DIAG_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case DiagAngetLogLevelType::DIAG_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case DiagAngetLogLevelType::DIAG_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case DiagAngetLogLevelType::DIAG_OFF:
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
        DiagAngetLogLevelType appLogLevel = DiagAngetLogLevelType::DIAG_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20  //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = DiagParseLogLevel(static_cast<int32_t> (appLogLevel));
        hozon::netaos::log::InitLogging(
            appId,
            appDescription,
            applevel,
            outputMode,
            directoryPath,
            maxLogFileNum,
            maxSizeOfLogFile
        );
    }

    // context regist diagserver
    void CreateLogger(const std::string ctxId)
    {
        const hozon::netaos::log::LogLevel level = DiagParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static DiagAgentLogger& GetInstance()
    {
        static DiagAgentLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> DiagGetLogger() const
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
    DiagAgentLogger(const DiagAgentLogger&);
    DiagAgentLogger& operator=(const DiagAgentLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(DiagAngetLogLevelType::DIAG_INFO);
};

#define DGA_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define DGA_CRITICAL (DiagAgentLogger::GetInstance().DiagGetLogger()->LogCritical() << DGA_HEAD)
#define DGA_ERROR (DiagAgentLogger::GetInstance().DiagGetLogger()->LogError() << DGA_HEAD)
#define DGA_WARN (DiagAgentLogger::GetInstance().DiagGetLogger()->LogWarn() << DGA_HEAD)
#define DGA_INFO (DiagAgentLogger::GetInstance().DiagGetLogger()->LogInfo() << DGA_HEAD)
#define DGA_DEBUG (DiagAgentLogger::GetInstance().DiagGetLogger()->LogDebug() << DGA_HEAD)
#define DGA_TRACE (DiagAgentLogger::GetInstance().DiagGetLogger()->LogTrace() << DGA_HEAD)

}  // namespace diag_agent
}  // namespace diag
}  // namespace netaos
}  // namespace hozon

#endif  // end of DIAG_AGENT_LOGGER_H_
// end of file
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: diag server loger
 */

#ifndef DIAG_SERVER_LOGGER_H_
#define DIAG_SERVER_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"
#include "diag/common/include/to_string.h"

namespace hozon {
namespace netaos {
namespace diag {

/* Fm log class */
class DiagServerLogger
{
public:
    DiagServerLogger() : logger_(nullptr) {};
    virtual ~DiagServerLogger() {};

    enum class DiagLogLevelType {
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
        const auto type = static_cast<DiagLogLevelType>(logLevel);
        switch (type) {
            case DiagLogLevelType::DIAG_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case DiagLogLevelType::DIAG_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case DiagLogLevelType::DIAG_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case DiagLogLevelType::DIAG_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case DiagLogLevelType::DIAG_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case DiagLogLevelType::DIAG_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case DiagLogLevelType::DIAG_OFF:
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
        DiagLogLevelType appLogLevel = DiagLogLevelType::DIAG_INFO, //the log level of application
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
            maxSizeOfLogFile,
            true
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

    static DiagServerLogger& GetInstance()
    {
        static DiagServerLogger instance;
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
    DiagServerLogger(const DiagServerLogger&);
    DiagServerLogger& operator=(const DiagServerLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(DiagLogLevelType::DIAG_INFO);
};

#define DG_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define DG_CRITICAL (DiagServerLogger::GetInstance().DiagGetLogger()->LogCritical() << DG_HEAD)
#define DG_ERROR (DiagServerLogger::GetInstance().DiagGetLogger()->LogError() << DG_HEAD)
#define DG_WARN (DiagServerLogger::GetInstance().DiagGetLogger()->LogWarn() << DG_HEAD)
#define DG_INFO (DiagServerLogger::GetInstance().DiagGetLogger()->LogInfo() << DG_HEAD)
#define DG_DEBUG (DiagServerLogger::GetInstance().DiagGetLogger()->LogDebug() << DG_HEAD)
#define DG_TRACE (DiagServerLogger::GetInstance().DiagGetLogger()->LogTrace() << DG_HEAD)

}  // namespace diag
}  // namespace netaos
}  // namespace hozon

#endif  // end of DIAG_SERVER_LOGGER_H_
// end of file

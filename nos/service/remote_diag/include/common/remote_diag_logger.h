/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: remote diag loger
 */

#ifndef REMOTE_DIAG_LOGGER_H_
#define REMOTE_DIAG_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"
#include "remote_diag/include/common/to_string.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

class RemoteDiagLogger
{
public:
    RemoteDiagLogger() : logger_(nullptr) {};
    virtual ~RemoteDiagLogger() {};

    enum class RemoteDiagLogLevelType {
        REMOTE_DIAG_TRACE = 0,
        REMOTE_DIAG_DEBUG = 1,
        REMOTE_DIAG_INFO = 2,
        REMOTE_DIAG_WARN = 3,
        REMOTE_DIAG_ERROR = 4,
        REMOTE_DIAG_CRITICAL = 5,
        REMOTE_DIAG_OFF = 6
    };


    hozon::netaos::log::LogLevel DiagParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<RemoteDiagLogLevelType>(logLevel);
        switch (type) {
            case RemoteDiagLogLevelType::REMOTE_DIAG_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case RemoteDiagLogLevelType::REMOTE_DIAG_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case RemoteDiagLogLevelType::REMOTE_DIAG_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case RemoteDiagLogLevelType::REMOTE_DIAG_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case RemoteDiagLogLevelType::REMOTE_DIAG_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case RemoteDiagLogLevelType::REMOTE_DIAG_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case RemoteDiagLogLevelType::REMOTE_DIAG_OFF:
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
        RemoteDiagLogLevelType appLogLevel = RemoteDiagLogLevelType::REMOTE_DIAG_INFO, //the log level of application
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

    static RemoteDiagLogger& GetInstance()
    {
        static RemoteDiagLogger instance;
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
    RemoteDiagLogger(const RemoteDiagLogger&);
    RemoteDiagLogger& operator=(const RemoteDiagLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(RemoteDiagLogLevelType::REMOTE_DIAG_INFO);
};

#define DGR_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define DGR_CRITICAL (RemoteDiagLogger::GetInstance().DiagGetLogger()->LogCritical() << DGR_HEAD)
#define DGR_ERROR (RemoteDiagLogger::GetInstance().DiagGetLogger()->LogError() << DGR_HEAD)
#define DGR_WARN (RemoteDiagLogger::GetInstance().DiagGetLogger()->LogWarn() << DGR_HEAD)
#define DGR_INFO (RemoteDiagLogger::GetInstance().DiagGetLogger()->LogInfo() << DGR_HEAD)
#define DGR_DEBUG (RemoteDiagLogger::GetInstance().DiagGetLogger()->LogDebug() << DGR_HEAD)
#define DGR_TRACE (RemoteDiagLogger::GetInstance().DiagGetLogger()->LogTrace() << DGR_HEAD)

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon

#endif  // end of REMOTE_DIAG_LOGGER_H_
// end of file

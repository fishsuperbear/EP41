/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: phm server loger
 */

#ifndef PHM_SERVER_LOGGER_H_
#define PHM_SERVER_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"
#include "phm_server/include/common/to_string.h"

namespace hozon {
namespace netaos {
namespace phm_server {

/* Phm log class */
class PHMServerLogger
{
public:
    PHMServerLogger() : logger_(nullptr) {};
    virtual ~PHMServerLogger() {};

    enum class PHMLogLevelType {
        PHM_VERBOSE = 0,
        PHM_DEBUG = 1,
        PHM_INFO = 2,
        PHM_WARN = 3,
        PHM_ERROR = 4,
        PHM_FATAL = 5,
        PHM_OFF = 6
    };


    hozon::netaos::log::LogLevel PHMParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<PHMLogLevelType>(logLevel);
        switch (type) {
            case PHMLogLevelType::PHM_VERBOSE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case PHMLogLevelType::PHM_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case PHMLogLevelType::PHM_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case PHMLogLevelType::PHM_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case PHMLogLevelType::PHM_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case PHMLogLevelType::PHM_FATAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case PHMLogLevelType::PHM_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "phm_server",  // the log id of application
        std::string appDescription = "phm_server application", // the log id of application
        PHMLogLevelType appLogLevel = PHMLogLevelType::PHM_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = PHMParseLogLevel(static_cast<int32_t> (appLogLevel));
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
        const hozon::netaos::log::LogLevel level = PHMParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static PHMServerLogger& GetInstance()
    {
        static PHMServerLogger instance;
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
    PHMServerLogger(const PHMServerLogger&);
    PHMServerLogger& operator=(const PHMServerLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(PHMLogLevelType::PHM_INFO);
};

#define PHMS_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define PHMS_FATAL (PHMServerLogger::GetInstance().DiagGetLogger()->LogFatal() << PHMS_HEAD)
#define PHMS_ERROR (PHMServerLogger::GetInstance().DiagGetLogger()->LogError() << PHMS_HEAD)
#define PHMS_WARN (PHMServerLogger::GetInstance().DiagGetLogger()->LogWarn() << PHMS_HEAD)
#define PHMS_INFO (PHMServerLogger::GetInstance().DiagGetLogger()->LogInfo() << PHMS_HEAD)
#define PHMS_DEBUG (PHMServerLogger::GetInstance().DiagGetLogger()->LogDebug() << PHMS_HEAD)
#define PHMS_VERBOSE (PHMServerLogger::GetInstance().DiagGetLogger()->LogVerbose() << PHMS_HEAD)

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon

#endif  // end of PHM_SERVER_LOGGER_H_
// end of file

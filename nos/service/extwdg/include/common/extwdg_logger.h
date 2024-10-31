/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: extwdg loger
 */

#ifndef EW_SERVER_LOGGER_H_
#define EW_SERVER_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"
#include "extwdg/include/common/to_string.h"

namespace hozon {
namespace netaos {
namespace extwdg {

/* Ew log class */
class EWServerLogger
{
public:
    EWServerLogger() : logger_(nullptr) {};
    virtual ~EWServerLogger() {};

    enum class EWLogLevelType {
        EW_VERBOSE = 0,
        EW_DEBUG = 1,
        EW_INFO = 2,
        EW_WARN = 3,
        EW_ERROR = 4,
        EW_FATAL = 5,
        EW_OFF = 6
    };


    hozon::netaos::log::LogLevel EWParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<EWLogLevelType>(logLevel);
        switch (type) {
            case EWLogLevelType::EW_VERBOSE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case EWLogLevelType::EW_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case EWLogLevelType::EW_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case EWLogLevelType::EW_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case EWLogLevelType::EW_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case EWLogLevelType::EW_FATAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case EWLogLevelType::EW_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "extwdg",  // the log id of application
        std::string appDescription = "ewtwdg application", // the log id of application
        EWLogLevelType appLogLevel = EWLogLevelType::EW_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = EWParseLogLevel(static_cast<int32_t> (appLogLevel));
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

    // context regist EWserver
    void CreateLogger(const std::string ctxId)
    {
        const hozon::netaos::log::LogLevel level = EWParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static EWServerLogger& GetInstance()
    {
        static EWServerLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> EWGetLogger() const
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
    EWServerLogger(const EWServerLogger&);
    EWServerLogger& operator=(const EWServerLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(EWLogLevelType::EW_INFO);
};

#define EW_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define EW_FATAL (EWServerLogger::GetInstance().EWGetLogger()->LogFatal() << EW_HEAD)
#define EW_ERROR (EWServerLogger::GetInstance().EWGetLogger()->LogError() << EW_HEAD)
#define EW_WARN (EWServerLogger::GetInstance().EWGetLogger()->LogWarn() << EW_HEAD)
#define EW_INFO (EWServerLogger::GetInstance().EWGetLogger()->LogInfo() << EW_HEAD)
#define EW_DEBUG (EWServerLogger::GetInstance().EWGetLogger()->LogDebug() << EW_HEAD)
#define EW_VERBOSE (EWServerLogger::GetInstance().EWGetLogger()->LogVerbose() << EW_HEAD)

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon

#endif  // end of EW_SERVER_LOGGER_H_
// end of file

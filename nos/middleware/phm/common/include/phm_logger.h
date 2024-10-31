/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip loger
 */

#ifndef PHM_LOGGER_H_
#define PHM_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace phm {

class PhmLogger
{
public:
    PhmLogger() {
        // char *env = getenv("PHM_LOG_OPEN");
        // if (env != NULL) {
        //     PHM_LOG_OPEN = std::atoi(env);
        // }
    };
    virtual ~PhmLogger() {};

    enum class PhmLogLevelType {
        PHM_TRACE = 0,
        PHM_DEBUG = 1,
        PHM_INFO = 2,
        PHM_WARN = 3,
        PHM_ERROR = 4,
        PHM_CRITICAL = 5,
        PHM_OFF = 6
    };

    hozon::netaos::log::LogLevel DiagParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<PhmLogLevelType>(logLevel);
        switch (type) {
            case PhmLogLevelType::PHM_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case PhmLogLevelType::PHM_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case PhmLogLevelType::PHM_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case PhmLogLevelType::PHM_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case PhmLogLevelType::PHM_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case PhmLogLevelType::PHM_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case PhmLogLevelType::PHM_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    void CreateLogger(const std::string ctxId)
    {
        const hozon::netaos::log::LogLevel level = DiagParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static PhmLogger& Instance()
    {
        static PhmLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> PhmGetLogger() const
    {
        return logger_;
        // if (PHM_LOG_OPEN) {
        //     return logger_;
        // }
        // else {
        //     return nullptr;
        // }
    }

    int32_t GetLogLevel()
    {
        return level_;
    }

    void SetLogLevel(int32_t level)
    {
        level_ = level;
    }

    int32_t GetLogMode()
    {
        return mode_;
    }

    void SetLogMode(int32_t mode)
    {
        mode_ = mode;
    }

private:
    PhmLogger(const PhmLogger&);
    PhmLogger& operator=(const PhmLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
private:
    int32_t level_ = static_cast<int32_t>(PhmLogLevelType::PHM_WARN);
    int32_t mode_ = static_cast<int32_t>(netaos::log::HZ_LOG2FILE);
    // int PHM_LOG_OPEN {0};
};

#define PHM_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define PHM_FATAL if(PhmLogger::Instance().PhmGetLogger())PhmLogger::Instance().PhmGetLogger()->LogFatal() << PHM_HEAD
#define PHM_ERROR if(PhmLogger::Instance().PhmGetLogger())PhmLogger::Instance().PhmGetLogger()->LogError() << PHM_HEAD
#define PHM_WARN if(PhmLogger::Instance().PhmGetLogger())PhmLogger::Instance().PhmGetLogger()->LogWarn() << PHM_HEAD
#define PHM_INFO if(PhmLogger::Instance().PhmGetLogger())PhmLogger::Instance().PhmGetLogger()->LogInfo() << PHM_HEAD
#define PHM_DEBUG if(PhmLogger::Instance().PhmGetLogger())PhmLogger::Instance().PhmGetLogger()->LogDebug() << PHM_HEAD
#define PHM_TRACE if(PhmLogger::Instance().PhmGetLogger())PhmLogger::Instance().PhmGetLogger()->LogTrace() << PHM_HEAD


// #define PHM_HEAD "\npid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
// #define PHM_FATAL std::cout << PHM_HEAD
// #define PHM_ERROR std::cout << PHM_HEAD
// #define PHM_WARN  std::cout << PHM_HEAD
// #define PHM_INFO  std::cout << PHM_HEAD
// #define PHM_DEBUG   std::cout << PHM_HEAD
// #define PHM_TRACE std::cout << PHM_HEAD

}  // namespace phm
}  // namespace netaos
}  // namespace hozon

#endif  // end of PHM_LOGGER_H_
// end of file

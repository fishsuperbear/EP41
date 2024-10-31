/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip loger
 */

#ifndef DOSOMEIP_LOGGER_H_
#define DOSOMEIP_LOGGER_H_

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

#include "log/include/logging.h"
#include "diag/common/include/to_string.h"

namespace hozon {
namespace netaos {
namespace diag {

class DoSomeIPLogger
{
public:
    DoSomeIPLogger() {
        // char *env = getenv("DOSOMEIP_LOG_OPEN");
        // if (env != NULL) {
        //     DOSOMEIP_LOG_OPEN = std::atoi(env);
        // }
    };
    virtual ~DoSomeIPLogger() {};

    enum class DoSomeIPLogLevelType {
        DOSOMEIP_TRACE = 0,
        DOSOMEIP_DEBUG = 1,
        DOSOMEIP_INFO = 2,
        DOSOMEIP_WARN = 3,
        DOSOMEIP_ERROR = 4,
        DOSOMEIP_CRITICAL = 5,
        DOSOMEIP_OFF = 6
    };

    hozon::netaos::log::LogLevel DiagParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<DoSomeIPLogLevelType>(logLevel);
        switch (type) {
            case DoSomeIPLogLevelType::DOSOMEIP_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case DoSomeIPLogLevelType::DOSOMEIP_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case DoSomeIPLogLevelType::DOSOMEIP_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case DoSomeIPLogLevelType::DOSOMEIP_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case DoSomeIPLogLevelType::DOSOMEIP_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case DoSomeIPLogLevelType::DOSOMEIP_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case DoSomeIPLogLevelType::DOSOMEIP_OFF:
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
        DoSomeIPLogLevelType appLogLevel = DoSomeIPLogLevelType::DOSOMEIP_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
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

    void CreateLogger(const std::string ctxId)
    {
        const hozon::netaos::log::LogLevel level = DiagParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static DoSomeIPLogger& GetInstance()
    {
        static DoSomeIPLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> DoSomeIPGetLogger() const
    {
        // if (DOSOMEIP_LOG_OPEN) {
        //     return logger_;
        // }
        // else {
        //     return nullptr;
        // }
        return logger_;
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
    DoSomeIPLogger(const DoSomeIPLogger&);
    DoSomeIPLogger& operator=(const DoSomeIPLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger> logger_{nullptr};
private:
    int32_t level_ = static_cast<int32_t>(DoSomeIPLogLevelType::DOSOMEIP_INFO);
    int32_t mode_ = static_cast<int32_t>(netaos::log::HZ_LOG2FILE);
    // int DOSOMEIP_LOG_OPEN {0};
};


#define DS_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define DS_CRITICAL if(DoSomeIPLogger::GetInstance().DoSomeIPGetLogger())DoSomeIPLogger::GetInstance().DoSomeIPGetLogger()->LogCritical() << DS_HEAD << hozon::netaos::log::FROM_HERE
#define DS_ERROR if(DoSomeIPLogger::GetInstance().DoSomeIPGetLogger())DoSomeIPLogger::GetInstance().DoSomeIPGetLogger()->LogError() << DS_HEAD << hozon::netaos::log::FROM_HERE
#define DS_WARN if(DoSomeIPLogger::GetInstance().DoSomeIPGetLogger())DoSomeIPLogger::GetInstance().DoSomeIPGetLogger()->LogWarn() << DS_HEAD << hozon::netaos::log::FROM_HERE
#define DS_INFO if(DoSomeIPLogger::GetInstance().DoSomeIPGetLogger())DoSomeIPLogger::GetInstance().DoSomeIPGetLogger()->LogInfo() << DS_HEAD << hozon::netaos::log::FROM_HERE
#define DS_DEBUG if(DoSomeIPLogger::GetInstance().DoSomeIPGetLogger())DoSomeIPLogger::GetInstance().DoSomeIPGetLogger()->LogDebug() << DS_HEAD << hozon::netaos::log::FROM_HERE
#define DS_TRACE if(DoSomeIPLogger::GetInstance().DoSomeIPGetLogger())DoSomeIPLogger::GetInstance().DoSomeIPGetLogger()->LogTrace() << DS_HEAD << hozon::netaos::log::FROM_HERE

}  // namespace diag
}  // namespace netaos
}  // namespace hozon

#endif  // end of DOSOMEIP_LOGGER_H_
// end of file

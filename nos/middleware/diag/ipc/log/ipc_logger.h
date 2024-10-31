/*
#pragma once

#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace diag {

class IPCLogger
{
public:
    IPCLogger() {
        // char *env = getenv("IPC_LOG_OPEN");
        // if (env != NULL) {
        //     IPC_LOG_OPEN = std::atoi(env);
        // }
    };
    virtual ~IPCLogger() {};

    enum class IPCLogLevelType {
        IPC_TRACE = 0,
        IPC_DEBUG = 1,
        IPC_INFO = 2,
        IPC_WARN = 3,
        IPC_ERROR = 4,
        IPC_CRITICAL = 5,
        IPC_OFF = 6
    };

    hozon::netaos::log::LogLevel DiagParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<IPCLogLevelType>(logLevel);
        switch (type) {
            case IPCLogLevelType::IPC_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case IPCLogLevelType::IPC_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case IPCLogLevelType::IPC_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case IPCLogLevelType::IPC_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case IPCLogLevelType::IPC_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case IPCLogLevelType::IPC_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case IPCLogLevelType::IPC_OFF:
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

    static IPCLogger& GetInstance()
    {
        static IPCLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> IPCGetLogger() const
    {
        // if (IPC_LOG_OPEN) {
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
    IPCLogger(const IPCLogger&);
    IPCLogger& operator=(const IPCLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger> logger_ { hozon::netaos::log::CreateLogger("IPC", "diag_ipc",
                        hozon::netaos::log::LogLevel::kInfo) };
private:
    int32_t level_ = static_cast<int32_t>(IPCLogLevelType::IPC_WARN);
    int32_t mode_ = static_cast<int32_t>(netaos::log::HZ_LOG2FILE);
};


#define IPC_HEAD "pid: " << (long int)syscall(__NR_getpid) <<  " tid: " << (long int)syscall(__NR_gettid) << " | "
#define IPC_CRITICAL if(IPCLogger::GetInstance().IPCGetLogger())IPCLogger::GetInstance().IPCGetLogger()->LogCritical() << IPC_HEAD << hozon::netaos::log::FROM_HERE
#define IPC_ERROR if(IPCLogger::GetInstance().IPCGetLogger())IPCLogger::GetInstance().IPCGetLogger()->LogError() << IPC_HEAD << hozon::netaos::log::FROM_HERE
#define IPC_WARN if(IPCLogger::GetInstance().IPCGetLogger())IPCLogger::GetInstance().IPCGetLogger()->LogWarn() << IPC_HEAD << hozon::netaos::log::FROM_HERE
#define IPC_INFO if(IPCLogger::GetInstance().IPCGetLogger())IPCLogger::GetInstance().IPCGetLogger()->LogInfo() << IPC_HEAD << hozon::netaos::log::FROM_HERE
#define IPC_DEBUG if(IPCLogger::GetInstance().IPCGetLogger())IPCLogger::GetInstance().IPCGetLogger()->LogDebug() << IPC_HEAD << hozon::netaos::log::FROM_HERE
#define IPC_TRACE if(IPCLogger::GetInstance().IPCGetLogger())IPCLogger::GetInstance().IPCGetLogger()->LogTrace() << IPC_HEAD << hozon::netaos::log::FROM_HERE

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
*/

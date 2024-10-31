/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: execution manager loger
 */

#ifndef EM_LOGGER_H_
#define EM_LOGGER_H_

#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace em {

using namespace std;

class EMLogger
{
public:
    EMLogger() : logger_(nullptr) {};
    virtual ~EMLogger() {};

    enum LogLevelType {
        LOG_LEVEL_TRACE = 0,
        LOG_LEVEL_DEBUG = 1,
        LOG_LEVEL_INFO = 2,
        LOG_LEVEL_WARN = 3,
        LOG_LEVEL_ERROR = 4,
        LOG_LEVEL_CRITICAL = 5,
        LOG_LEVEL_OFF = 6
    };


    hozon::netaos::log::LogLevel ParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<LogLevelType>(logLevel);
        switch (type) {
            case LogLevelType::LOG_LEVEL_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case LogLevelType::LOG_LEVEL_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case LogLevelType::LOG_LEVEL_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case LogLevelType::LOG_LEVEL_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case LogLevelType::LOG_LEVEL_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case LogLevelType::LOG_LEVEL_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case LogLevelType::LOG_LEVEL_OFF:
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
        LogLevelType appLogLevel = LogLevelType::LOG_LEVEL_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
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
            maxSizeOfLogFile
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

    static EMLogger& GetInstance()
    {
        static EMLogger instance;
        return instance;
    }

    int32_t getLogLevel()
    {
        return level_;
    }

    void setLogLevel(int32_t level)
    {
        level_ = level;
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const
    {
        return logger_;
    }

private:
    EMLogger(const EMLogger&);
    EMLogger& operator=(const EMLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(LogLevelType::LOG_LEVEL_INFO);
};

template <class T>
static std::string ToString(std::vector<T> t, std::ios_base & (*f)(std::ios_base&), int n = 0)
{
    if (t.size() <= 0) {
        return "";
    }

    std::ostringstream oss;
    int typesize = sizeof(t[0]);
    for (uint i = 0; i < t.size();) {
        if (n) {
            oss << std::setw(n) << std::setfill('0');
        }

        if (1 == typesize) {
            uint8_t item = static_cast<uint8_t>(t[i]);
            oss << f << static_cast<uint16_t>(item);
        }
        else {
            oss << f << t[i];
        }

        ++i;

        if (i < t.size()) {
            oss << " ";
        }
    }

    return oss.str();
}

#define UM_UINT8_VEC_TO_HEX_STRING(vec) ToString<uint8_t>(vec, std::hex, 2)

// #define EM_HEAD getpid() << " " << (long int)syscall(__NR_gettid) << " " <<__FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "
#define EM_HEAD getpid() << " " << (long int)syscall(__NR_gettid) << " " <<__FUNCTION__ << " | "
#define _LOG_CRITICAL (EMLogger::GetInstance().GetLogger()->LogCritical() << EM_HEAD)
#define _LOG_ERROR (EMLogger::GetInstance().GetLogger()->LogError() << EM_HEAD)
#define _LOG_WARN (EMLogger::GetInstance().GetLogger()->LogWarn() << EM_HEAD)
#define _LOG_INFO (EMLogger::GetInstance().GetLogger()->LogInfo() << EM_HEAD)
#define _LOG_DEBUG (EMLogger::GetInstance().GetLogger()->LogDebug() << EM_HEAD)
#define _LOG_TRACE (EMLogger::GetInstance().GetLogger()->LogTrace() << EM_HEAD)
#define BR (EMLogger::GetInstance().GetLogger()->LogTrace())

#define EM_LOG_D(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    EM_DEBUG << szTempLogInfo;               \
  } while (0)

#define EM_LOG_E(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    EM_ERROR << szTempLogInfo;               \
  } while (0)

#define EM_LOG_W(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    EM_WARN << szTempLogInfo;                \
  } while (0)

#define EM_LOG_I(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    EM_INFO << szTempLogInfo;                \
  } while (0)

}  // namespace em
}  // netaos
}  // namespace hozon

#endif  // end of EM_LOGGER_H_
// end of file

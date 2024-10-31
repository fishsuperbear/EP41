/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  DocanLog defination Header
 */


#ifndef DOCAN_LOG_H_
#define DOCAN_LOG_H_

#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <vector>
#include <sys/syscall.h>
#include "log/include/logging.h"


namespace hozon {
namespace netaos {
namespace diag {

/* Fm log class */
class DocanLogger
{
public:
    DocanLogger() : logger_(nullptr) {};
    virtual ~DocanLogger() {};

    enum class DocanLogLevelType {
        LOG_LEVEL_TRACE = 0,
        LOG_LEVEL_DEBUG = 1,
        LOG_LEVEL_INFO = 2,
        LOG_LEVEL_WARN = 3,
        LOG_LEVEL_ERROR = 4,
        LOG_LEVEL_CRITICAL = 5,
        LOG_LEVEL_OFF = 6
    };


    hozon::netaos::log::LogLevel DocanParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<DocanLogLevelType>(logLevel);
        switch (type) {
            case DocanLogLevelType::LOG_LEVEL_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case DocanLogLevelType::LOG_LEVEL_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case DocanLogLevelType::LOG_LEVEL_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case DocanLogLevelType::LOG_LEVEL_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case DocanLogLevelType::LOG_LEVEL_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case DocanLogLevelType::LOG_LEVEL_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case DocanLogLevelType::LOG_LEVEL_OFF:
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
        DocanLogLevelType appLogLevel = DocanLogLevelType::LOG_LEVEL_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = DocanParseLogLevel(static_cast<int32_t> (appLogLevel));
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
        const hozon::netaos::log::LogLevel level = DocanParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static DocanLogger& GetInstance()
    {
        static DocanLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> DocanGetLogger() const
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
    DocanLogger(const DocanLogger&);
    DocanLogger& operator=(const DocanLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(DocanLogLevelType::LOG_LEVEL_INFO);
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

#define DOCAN_UINT8_VEC_TO_HEX_STRING(vec) ToString<uint8_t>(vec, std::hex, 2)

#define DOCAN_HEAD getpid() << " " << (long int)syscall(__NR_gettid) << " " <<__FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "
#define DOCAN_CRITICAL (DocanLogger::GetInstance().DocanGetLogger()->LogCritical() << DOCAN_HEAD)
#define DOCAN_ERROR (DocanLogger::GetInstance().DocanGetLogger()->LogError() << DOCAN_HEAD)
#define DOCAN_WARN (DocanLogger::GetInstance().DocanGetLogger()->LogWarn() << DOCAN_HEAD)
#define DOCAN_INFO (DocanLogger::GetInstance().DocanGetLogger()->LogInfo() << DOCAN_HEAD)
#define DOCAN_DEBUG (DocanLogger::GetInstance().DocanGetLogger()->LogDebug() << DOCAN_HEAD)
#define DOCAN_TRACE (DocanLogger::GetInstance().DocanGetLogger()->LogTrace() << DOCAN_HEAD)

#define DOCAN_LOG_D(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    DOCAN_DEBUG << szTempLogInfo;               \
  } while (0)

#define DOCAN_LOG_E(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    DOCAN_ERROR << szTempLogInfo;               \
  } while (0)

#define DOCAN_LOG_W(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    DOCAN_WARN << szTempLogInfo;                \
  } while (0)

#define DOCAN_LOG_I(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    DOCAN_INFO << szTempLogInfo;                \
  } while (0)

}  // namespace diag
}  // namespace netaos
}  // namespace hozon

#endif  // end of DOCAN_LOG_H_
// end of file

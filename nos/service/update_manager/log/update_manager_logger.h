/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: update manager loger
 */

#ifndef UPDATA_MANAGER_LOGGER_H_
#define UPDATA_MANAGER_LOGGER_H_

#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "log/include/logging.h"
#include <vector>


namespace hozon {
namespace netaos {
namespace update {

/* Fm log class */
class UpdateManagerLogger
{
public:
    UpdateManagerLogger() : logger_(nullptr) {};
    virtual ~UpdateManagerLogger() {};

    enum class UpdateLogLevelType {
        LOG_LEVEL_TRACE = 0,
        LOG_LEVEL_DEBUG = 1,
        LOG_LEVEL_INFO = 2,
        LOG_LEVEL_WARN = 3,
        LOG_LEVEL_ERROR = 4,
        LOG_LEVEL_CRITICAL = 5,
        LOG_LEVEL_OFF = 6
    };


    hozon::netaos::log::LogLevel UpdateParseLogLevel(const int32_t logLevel)
    {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<UpdateLogLevelType>(logLevel);
        switch (type) {
            case UpdateLogLevelType::LOG_LEVEL_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case UpdateLogLevelType::LOG_LEVEL_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case UpdateLogLevelType::LOG_LEVEL_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case UpdateLogLevelType::LOG_LEVEL_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case UpdateLogLevelType::LOG_LEVEL_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case UpdateLogLevelType::LOG_LEVEL_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case UpdateLogLevelType::LOG_LEVEL_OFF:
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
        UpdateLogLevelType appLogLevel = UpdateLogLevelType::LOG_LEVEL_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = UpdateParseLogLevel(static_cast<int32_t> (appLogLevel));
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
        const hozon::netaos::log::LogLevel level = UpdateParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static UpdateManagerLogger& GetInstance()
    {
        static UpdateManagerLogger instance;
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
    UpdateManagerLogger(const UpdateManagerLogger&);
    UpdateManagerLogger& operator=(const UpdateManagerLogger&);

public:
    std::shared_ptr<hozon::netaos::log::Logger>  logger_;
private:
    int32_t level_ = static_cast<int32_t>(UpdateLogLevelType::LOG_LEVEL_INFO);
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

#define UM_HEAD getpid() << " " << (long int)syscall(__NR_gettid) << " " <<__FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "
#define UM_CRITICAL (UpdateManagerLogger::GetInstance().DiagGetLogger()->LogCritical() << UM_HEAD)
#define UM_ERROR (UpdateManagerLogger::GetInstance().DiagGetLogger()->LogError() << UM_HEAD)
#define UM_WARN (UpdateManagerLogger::GetInstance().DiagGetLogger()->LogWarn() << UM_HEAD)
#define UM_INFO (UpdateManagerLogger::GetInstance().DiagGetLogger()->LogInfo() << UM_HEAD)
#define UM_DEBUG (UpdateManagerLogger::GetInstance().DiagGetLogger()->LogDebug() << UM_HEAD)
#define UM_TRACE (UpdateManagerLogger::GetInstance().DiagGetLogger()->LogTrace() << UM_HEAD)

#define UPDATE_LOG_D(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    UM_DEBUG << szTempLogInfo;               \
  } while (0)

#define UPDATE_LOG_E(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    UM_ERROR << szTempLogInfo;               \
  } while (0)

#define UPDATE_LOG_W(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    UM_WARN << szTempLogInfo;                \
  } while (0)

#define UPDATE_LOG_I(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    UM_INFO << szTempLogInfo;                \
  } while (0)

}  // namespace update
}  // namespace netaos
}  // namespace hozon

#endif  // end of UPDATA_MANAGER_LOGGER_H_
// end of file

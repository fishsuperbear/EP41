/**
 * Copyright @ 2021 - 2023 Hozon Auto Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * Hozon Auto Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * @file  STLogDef.h
 * @brief Class of STLogDef
 */
#ifndef STLOGDEF_H
#define STLOGDEF_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "STLog.h"
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <vector>
#include <string>
#include "log/include/logging.h"




/* Fm log class */
class STTaskLogger
{
public:
    STTaskLogger() : logger_(nullptr) {};
    virtual ~STTaskLogger() {};

    enum class LogLevelType {
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

    static STTaskLogger& GetInstance()
    {
        static STTaskLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const
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
    STTaskLogger(const STTaskLogger&);
    STTaskLogger& operator=(const STTaskLogger&);

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

#define STTASK_UINT8_VEC_TO_HEX_STRING(vec) ToString<uint8_t>(vec, std::hex, 2)

#define STTASK_HEAD getpid() << " " << (long int)syscall(__NR_gettid) << " " <<__FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "
#define STTASK_CRITICAL (STTaskLogger::GetInstance().GetLogger()->LogCritical() << STTASK_HEAD)
#define STTASK_ERROR (STTaskLogger::GetInstance().GetLogger()->LogError() << STTASK_HEAD)
#define STTASK_WARN (STTaskLogger::GetInstance().GetLogger()->LogWarn() << STTASK_HEAD)
#define STTASK_INFO (STTaskLogger::GetInstance().GetLogger()->LogInfo() << STTASK_HEAD)
#define STTASK_DEBUG (STTaskLogger::GetInstance().GetLogger()->LogDebug() << STTASK_HEAD)
#define STTASK_TRACE (STTaskLogger::GetInstance().GetLogger()->LogTrace() << STTASK_HEAD)

#define STLOG_D(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    STTASK_DEBUG << szTempLogInfo;               \
  } while (0)

#define STLOG_E(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    STTASK_ERROR << szTempLogInfo;               \
  } while (0)

#define STLOG_W(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    STTASK_WARN << szTempLogInfo;                \
  } while (0)

#define STLOG_I(...)                             \
  do {                                              \
    char szTempLogInfo[1024] = {0};                 \
    snprintf(szTempLogInfo, 1024 - 1, __VA_ARGS__); \
    STTASK_INFO << szTempLogInfo;                \
  } while (0)


enum LOG_TYPE : uint8_t {
    LOG_VERBOSE = 'V',
    LOG_DEBUG   = 'D',
    LOG_INFO    = 'I',
    LOG_WARN    = 'W',
    LOG_ERROR   = 'E',
};


#define STTASK_LOG_TAG "STTK"
#define STTASK_LOG_SUB_TAG "task"

#ifndef STTASK_LOG_V
#   define STTASK_LOG_V(...) ((void)hozon::netaos::sttask::STLog::output(STTASK_LOG_TAG, STTASK_LOG_SUB_TAG, LOG_VERBOSE, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__))
#endif
#ifndef STTASK_LOG_D
#   define STTASK_LOG_D(...) ((void)hozon::netaos::sttask::STLog::output(STTASK_LOG_TAG, STTASK_LOG_SUB_TAG, LOG_DEBUG, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__))
#endif
#ifndef STTASK_LOG_I
#   define STTASK_LOG_I(...) ((void)hozon::netaos::sttask::STLog::output(STTASK_LOG_TAG, STTASK_LOG_SUB_TAG, LOG_INFO, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__))
#endif
#ifndef STTASK_LOG_W
#   define STTASK_LOG_W(...) ((void)hozon::netaos::sttask::STLog::output(STTASK_LOG_TAG, STTASK_LOG_SUB_TAG, LOG_WARN, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__))
#endif
#ifndef STTASK_LOG_E
#   define STTASK_LOG_E(...) ((void)hozon::netaos::sttask::STLog::output(STTASK_LOG_TAG, STTASK_LOG_SUB_TAG, LOG_ERROR, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__))
#endif

#endif /* STLOGDEF_H */
/* EOF */
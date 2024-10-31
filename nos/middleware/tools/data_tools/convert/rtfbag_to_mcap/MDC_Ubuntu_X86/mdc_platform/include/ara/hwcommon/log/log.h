/*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
* Description: Log class header
* Create: 2019-7-2
*/
#ifndef INC_ARA_GODEL_COMMON_LOG_H
#define INC_ARA_GODEL_COMMON_LOG_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <map>
#include "ara/hwcommon/log/log_stream_buffer.h"
#include "ara/hwcommon/log/common.h"

namespace ara    {
namespace godel  {
namespace common {
namespace log    {
struct LogLimitConfig {
    std::uint32_t logNumber = 0U;
    enum class LimitType:uint8_t {
        NOLIMIT = 0U,
        TIMELIMIT,
        CALLONCE,
    };
    LimitType limit = LimitType::NOLIMIT;
};
class BaseLog;
class Log;
class LogVariable {
public:
    static std::shared_ptr<LogVariable>& GetInstance() noexcept;
    void SetAppId(const std::string& id);
    std::shared_ptr<Log> GetLogInstance(const std::string& ctxId, const std::string& ctxDescription);
    std::mutex logMutex;
    ~LogVariable() = default;
private:
    LogVariable() = default;
    std::unordered_map<std::string, std::shared_ptr<Log>> logModules;
    // The mutex is used in aos-core and the macro is not impl beacause of the head file is public for user.
    std::mutex mutex_;
    std::string appId_;
};

class Log {
public:
    struct LogLimitResult {
        bool  logLimit;
        std::uint32_t logLimitCount;
    };
    struct LogRecordInfo {
        std::uint64_t startTime;
        std::uint32_t printLogCount;
    };
    static std::shared_ptr<Log> GetInstance(std::string const &ctxId, std::string const &ctxDescription);
    static std::shared_ptr<Log> GetLog(std::string const &ctxId);
    std::unordered_map<uint8_t, std::shared_ptr<BaseLog>> drivers {};
    Log(std::string const &ctxId, std::string const &ctxDescription);
    // Change the interface smoothly and delete it after adaptation

    LogStreamBuffer fatal(std::string const &logUUID = "",
            const LogLimitConfig& config = {0U, LogLimitConfig::LimitType::TIMELIMIT}) noexcept;
    LogStreamBuffer error(std::string const &logUUID = "",
            const LogLimitConfig& config = {0U, LogLimitConfig::LimitType::TIMELIMIT}) noexcept;
    LogStreamBuffer warn(std::string const &logUUID = "",
            const LogLimitConfig& config = {0U, LogLimitConfig::LimitType::TIMELIMIT}) noexcept;
    LogStreamBuffer info(std::string const &logUUID = "",
            const LogLimitConfig& config = {0U, LogLimitConfig::LimitType::TIMELIMIT}) noexcept;
    LogStreamBuffer debug() noexcept;
    LogStreamBuffer verbose() noexcept;
    static void InitLogging(std::string const &appId,
        std::string const &appDescription,
        LogLevel appDefLogLevel = LogLevel::VRTF_COMMON_LOG_WARN,
        LogMode appLogMode = LogMode::LOG_REMOTE,
        std::string const &directoryPath = "");
    static void InitLog(std::string const &appId, std::string const &ctxId, std::string const &ctxDescription);
    void ParseDriverAccordingEnvVar(int32_t index);
    void SetLogLevel(LogLevel lv) noexcept;
    LogLevel level;
    ~Log() {}
    std::string GetCtxId() const;
    bool IsValid() const noexcept;
    void IsInit(bool isInit) noexcept;
    bool IsInit() const noexcept;
private:
    static uint64_t GetCurrentMonotonicTime() noexcept;
    void ParseDriverType();
    Log::LogLimitResult LimitLogPrintOfUnitTime(std::string  const &logUUID, const LogLimitConfig& config);
    friend class LogStreamBuffer;
    std::string ctxId_;
    std::string ctxDescription_;
    bool containsAraLog_;
    Log(Log const &other) = default;
    Log& operator=(Log const &other) = default;
    bool isValid_;
    bool isInit_ {false};

    std::map<size_t, LogRecordInfo> logLimit_;
    std::shared_ptr<LogVariable> logVariable_;
};

template<typename T>
void StreamSplice(LogStreamBuffer &logStream, const T &t) noexcept
{
    logStream << t;
}

template<typename T, typename... Args>
void StreamSplice(LogStreamBuffer &logStream, const T &t, const Args&... rest) noexcept
{
    logStream << t;
    StreamSplice(logStream, rest...);
}

#define RTF_FATAL_LOG_SPR(logger, logUUID, number, ...)                                 \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_FATAL)) { \
            LogStreamBuffer stream{logger->fatal(logUUID, {number, LogLimitConfig::LimitType::TIMELIMIT})};      \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_ERROR_LOG_SPR(logger, logUUID, number, ...)                                 \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_ERROR)) { \
            LogStreamBuffer stream{logger->error(logUUID, {number, LogLimitConfig::LimitType::TIMELIMIT})};      \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_WARN_LOG_SPR(logger, logUUID, number, ...)                                  \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_WARN)) {  \
            LogStreamBuffer stream{logger->warn(logUUID, {number, LogLimitConfig::LimitType::TIMELIMIT})};       \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_INFO_LOG_SPR(logger, logUUID, number, ...)                                  \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_INFO)) {  \
            LogStreamBuffer stream{logger->info(logUUID, {number, LogLimitConfig::LimitType::TIMELIMIT})};       \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_FATAL_LOG(logger, ...)                                                      \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_FATAL)) { \
            LogStreamBuffer stream{logger->fatal()};                                    \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_ERROR_LOG(logger, ...)                                                      \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_ERROR)) { \
            LogStreamBuffer stream{logger->error()};                                    \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_WARN_LOG(logger, ...)                                                       \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_WARN)) {  \
            LogStreamBuffer stream{logger->warn()};                                     \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_INFO_LOG(logger, ...)                                                       \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_INFO)) {  \
            LogStreamBuffer stream{logger->info()};                                     \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_DEBUG_LOG(logger, ...)                                                      \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_DEBUG)) { \
            LogStreamBuffer stream{logger->debug()};                                    \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

#define RTF_VERBOSE_LOG(logger, ...)                                                    \
    do {                                                                                \
        using namespace ara::godel::common::log;                                        \
        if (logger != nullptr &&                                                        \
            LogStreamBuffer::CheckLogLevel(*logger, ara::godel::common::log::LogLevel::VRTF_COMMON_LOG_VERBOSE)) { \
            LogStreamBuffer stream{logger->verbose()};                                  \
            StreamSplice(stream, __VA_ARGS__);                                          \
        }                                                                               \
    } while (false)

}
}
}
}
#endif

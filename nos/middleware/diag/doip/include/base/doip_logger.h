/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip loger
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_LOGGER_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_LOGGER_H_

#include <unistd.h>
#include <sys/syscall.h>

#include <string>
#include <memory>

#include "log/include/logging.h"
#include "diag/common/include/to_string.h"

namespace hozon {
namespace netaos {
namespace diag {

class DoIPLogger {
 public:
    DoIPLogger() {}
    virtual ~DoIPLogger() {}

    enum class DoIPLogLevelType {
        DOIP_TRACE = 0,
        DOIP_DEBUG = 1,
        DOIP_INFO = 2,
        DOIP_WARN = 3,
        DOIP_ERROR = 4,
        DOIP_CRITICAL = 5,
        DOIP_OFF = 6
    };

    hozon::netaos::log::LogLevel DiagParseLogLevel(const int32_t logLevel) {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<DoIPLogLevelType>(logLevel);
        switch (type) {
            case DoIPLogLevelType::DOIP_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case DoIPLogLevelType::DOIP_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case DoIPLogLevelType::DOIP_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case DoIPLogLevelType::DOIP_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case DoIPLogLevelType::DOIP_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case DoIPLogLevelType::DOIP_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case DoIPLogLevelType::DOIP_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    void CreateLogger(const std::string ctxId) {
        const hozon::netaos::log::LogLevel level = DiagParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static DoIPLogger& GetInstance() {
        static DoIPLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> DoIPGetLogger() const {
        return logger_;
    }

    int32_t GetLogLevel() {
        return level_;
    }

    void SetLogLevel(int32_t level) {
        level_ = level;
    }

    int32_t GetLogMode() {
        return mode_;
    }

    void SetLogMode(int32_t mode) {
        mode_ = mode;
    }

 private:
    DoIPLogger(const DoIPLogger&);
    DoIPLogger& operator=(const DoIPLogger&);

 public:
    std::shared_ptr<hozon::netaos::log::Logger> logger_{nullptr};

 private:
    int32_t level_ = static_cast<int32_t>(DoIPLogLevelType::DOIP_INFO);
    int32_t mode_ = static_cast<int32_t>(netaos::log::HZ_LOG2FILE);
};


#define DOIP_HEAD "pid: " << (int32_t)syscall(__NR_getpid) <<  " tid: " << (int32_t)syscall(__NR_gettid) << " | "
#define DOIP_CRITICAL if (DoIPLogger::GetInstance().DoIPGetLogger())DoIPLogger::GetInstance().DoIPGetLogger()->LogCritical() << DOIP_HEAD
#define DOIP_ERROR if (DoIPLogger::GetInstance().DoIPGetLogger())DoIPLogger::GetInstance().DoIPGetLogger()->LogError() << DOIP_HEAD
#define DOIP_WARN if (DoIPLogger::GetInstance().DoIPGetLogger())DoIPLogger::GetInstance().DoIPGetLogger()->LogWarn() << DOIP_HEAD
#define DOIP_INFO if (DoIPLogger::GetInstance().DoIPGetLogger())DoIPLogger::GetInstance().DoIPGetLogger()->LogInfo() << DOIP_HEAD
#define DOIP_DEBUG if (DoIPLogger::GetInstance().DoIPGetLogger())DoIPLogger::GetInstance().DoIPGetLogger()->LogDebug() << DOIP_HEAD
#define DOIP_TRACE if (DoIPLogger::GetInstance().DoIPGetLogger())DoIPLogger::GetInstance().DoIPGetLogger()->LogTrace() << DOIP_HEAD


}  // namespace diag
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_LOGGER_H_

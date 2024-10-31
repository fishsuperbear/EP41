/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: 20000 loger
 */

#ifndef MIDDLEWARE_DIAG_DEVMTOOL_INCLUDE_BASE_DEVMTOOL_LOGGER_H_
#define MIDDLEWARE_DIAG_DEVMTOOL_INCLUDE_BASE_DEVMTOOL_LOGGER_H_

#include <unistd.h>
#include <sys/syscall.h>

#include <string>
#include <memory>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace tools {

class DevmToolLogger {
 public:
    DevmToolLogger() {}
    virtual ~DevmToolLogger() {}

    enum class DevmToolLogLevelType {
        DTOOL_TRACE = 0,
        DTOOL_DEBUG = 1,
        DTOOL_INFO = 2,
        DTOOL_WARN = 3,
        DTOOL_ERROR = 4,
        DTOOL_CRITICAL = 5,
        DTOOL_OFF = 6
    };

    hozon::netaos::log::LogLevel DiagParseLogLevel(const int32_t logLevel) {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<DevmToolLogLevelType>(logLevel);
        switch (type) {
            case DevmToolLogLevelType::DTOOL_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case DevmToolLogLevelType::DTOOL_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case DevmToolLogLevelType::DTOOL_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case DevmToolLogLevelType::DTOOL_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case DevmToolLogLevelType::DTOOL_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case DevmToolLogLevelType::DTOOL_CRITICAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case DevmToolLogLevelType::DTOOL_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "DevmTool",  // the log id of application
        std::string appDescription = "devm tool", // the log id of application
        DevmToolLogLevelType appLogLevel = DevmToolLogLevelType::DTOOL_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20  //the max size of each  log file , active when output log to file
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
            maxSizeOfLogFile,
            true
        );
    }
    void CreateLogger(const std::string ctxId) {
        const hozon::netaos::log::LogLevel level = DiagParseLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    static DevmToolLogger& GetInstance() {
        static DevmToolLogger instance;
        return instance;
    }

    std::shared_ptr<hozon::netaos::log::Logger> DevmToolGetLogger() const {
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
    DevmToolLogger(const DevmToolLogger&);
    DevmToolLogger& operator=(const DevmToolLogger&);

 public:
    std::shared_ptr<hozon::netaos::log::Logger> logger_{nullptr};

 private:
    int32_t level_ = static_cast<int32_t>(DevmToolLogLevelType::DTOOL_INFO);
    int32_t mode_ = static_cast<int32_t>(netaos::log::HZ_LOG2FILE);
};


#define DEVMTOOL_HEAD "pid: " << (int32_t)syscall(__NR_getpid) <<  " tid: " << (int32_t)syscall(__NR_gettid) << " | "
#define DEVMTOOL_CRITICAL if (DevmToolLogger::GetInstance().DevmToolGetLogger())DevmToolLogger::GetInstance().DevmToolGetLogger()->LogCritical() << DEVMTOOL_HEAD
#define DEVMTOOL_ERROR if (DevmToolLogger::GetInstance().DevmToolGetLogger())DevmToolLogger::GetInstance().DevmToolGetLogger()->LogError() << DEVMTOOL_HEAD
#define DEVMTOOL_WARN if (DevmToolLogger::GetInstance().DevmToolGetLogger())DevmToolLogger::GetInstance().DevmToolGetLogger()->LogWarn() << DEVMTOOL_HEAD
#define DEVMTOOL_INFO if (DevmToolLogger::GetInstance().DevmToolGetLogger())DevmToolLogger::GetInstance().DevmToolGetLogger()->LogInfo() << DEVMTOOL_HEAD
#define DEVMTOOL_DEBUG if (DevmToolLogger::GetInstance().DevmToolGetLogger())DevmToolLogger::GetInstance().DevmToolGetLogger()->LogDebug() << DEVMTOOL_HEAD
#define DEVMTOOL_TRACE if (DevmToolLogger::GetInstance().DevmToolGetLogger())DevmToolLogger::GetInstance().DevmToolGetLogger()->LogTrace() << DEVMTOOL_HEAD


}  // namespace tools
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_DIAG_DEVMTOOL_INCLUDE_BASE_DEVMTOOL_LOGGER_H_

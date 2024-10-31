/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: ota api logger
 */

#ifndef OTA_API_LOGGER_H_
#define OTA_API_LOGGER_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace otaapi {
/* ota_api log class */
class OtaApiLogger {
public:
    static OtaApiLogger& GetInstance() {
        static OtaApiLogger instance;
        return instance;
    }
    ~OtaApiLogger(){};
    enum class OTAAPILogLevelType { OTAAPI_TRACE = 0, OTAAPI_DEBUG = 1, OTAAPI_INFO = 2, OTAAPI_WARN = 3, OTAAPI_ERROR = 4, OTAAPI_FATAL = 5, OTAAPI_OFF = 6 };
    hozon::netaos::log::LogLevel OtaApiLogLevel(const int32_t logLevel) {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<OTAAPILogLevelType>(logLevel);
        switch (type) {
            case OTAAPILogLevelType::OTAAPI_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case OTAAPILogLevelType::OTAAPI_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case OTAAPILogLevelType::OTAAPI_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case OTAAPILogLevelType::OTAAPI_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case OTAAPILogLevelType::OTAAPI_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case OTAAPILogLevelType::OTAAPI_FATAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case OTAAPILogLevelType::OTAAPI_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "ota_api",  // the log id of application
        std::string appDescription = "ota_api application", // the log id of application
        OTAAPILogLevelType appLogLevel = OTAAPILogLevelType::OTAAPI_INFO, //the log level of application
        std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        std::string directoryPath = "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        std::uint32_t maxLogFileNum = 10, //the max number log file , active when output log to file
        std::uint64_t maxSizeOfLogFile = 20 //the max size of each  log file , active when output log to file
    )
    {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = OtaApiLogLevel(static_cast<int32_t> (appLogLevel));
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
        const hozon::netaos::log::LogLevel level = OtaApiLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

private:
    OtaApiLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
    int32_t level_ = static_cast<int32_t>(OTAAPILogLevelType::OTAAPI_INFO);
};

#define OTA_API_LOG_HEAD        \
    " pid:" << getpid() << " " \
            << "tid:" << (long int)syscall(__NR_gettid) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define OTA_API_LOG_CRITICAL hozon::netaos::otaapi::OtaApiLogger::GetInstance().GetLogger()->LogCritical() << OTA_API_LOG_HEAD
#define OTA_API_LOG_ERROR hozon::netaos::otaapi::OtaApiLogger::GetInstance().GetLogger()->LogError() << OTA_API_LOG_HEAD
#define OTA_API_LOG_WARN hozon::netaos::otaapi::OtaApiLogger::GetInstance().GetLogger()->LogWarn() << OTA_API_LOG_HEAD
#define OTA_API_LOG_INFO hozon::netaos::otaapi::OtaApiLogger::GetInstance().GetLogger()->LogInfo() << OTA_API_LOG_HEAD
#define OTA_API_LOG_DEBUG hozon::netaos::otaapi::OtaApiLogger::GetInstance().GetLogger()->LogDebug() << OTA_API_LOG_HEAD
#define OTA_API_LOG_TRACE hozon::netaos::otaapi::OtaApiLogger::GetInstance().GetLogger()->LogTrace() << OTA_API_LOG_HEAD

}  // namespace otaapi
}  // namespace netaos
}  // namespace hozon
#endif  // end of OTA_API_LOGGER_H_

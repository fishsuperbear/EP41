/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */

#ifndef DEVM_SERVER_LOGGER_H_
#define DEVM_SERVER_LOGGER_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace devm_server {
/* devm_server log class */
class DevmServerLogger {
   public:
    static DevmServerLogger& GetInstance() {
        static DevmServerLogger instance;
        return instance;
    }
    ~DevmServerLogger(){};
    enum class DEVMLogLevelType { DEVM_TRACE = 0, DEVM_DEBUG = 1, DEVM_INFO = 2, DEVM_WARN = 3, DEVM_ERROR = 4, DEVM_FATAL = 5, DEVM_OFF = 6 };
    hozon::netaos::log::LogLevel DevmLogLevel(const int32_t logLevel) {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<DEVMLogLevelType>(logLevel);
        switch (type) {
            case DEVMLogLevelType::DEVM_TRACE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case DEVMLogLevelType::DEVM_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case DEVMLogLevelType::DEVM_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case DEVMLogLevelType::DEVM_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case DEVMLogLevelType::DEVM_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case DEVMLogLevelType::DEVM_FATAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case DEVMLogLevelType::DEVM_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "devm_server",                                                                // the log id of application
                     std::string appDescription = "devm_server application",                                           // the log id of application
                     DEVMLogLevelType appLogLevel = DEVMLogLevelType::DEVM_INFO,                                      //the log level of application
                     std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE,  //the output log mode
                     std::string directoryPath = "/opt/usr/log/soc_log/",                                                                 //the log file directory, active when output log to file
                     std::uint32_t maxLogFileNum = 10,                                                                 //the max number log file , active when output log to file
                     std::uint64_t maxSizeOfLogFile = 20                                                               //the max size of each  log file , active when output log to file
    ) {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = DevmLogLevel(static_cast<int32_t>(appLogLevel));
        hozon::netaos::log::InitLogging(appId, appDescription, applevel, outputMode, directoryPath, maxLogFileNum, maxSizeOfLogFile);
    }

    // context regist diagserver
    void CreateLogger(const std::string ctxId) {
        const hozon::netaos::log::LogLevel level = DevmLogLevel(level_);
        std::string ctxIdView(ctxId.c_str());
        std::string ctxDescription(ctxId + " Loger");
        std::string ctxDescView(ctxDescription.c_str());
        auto logger = hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
        logger_ = logger;
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

   private:
    DevmServerLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
    int32_t level_ = static_cast<int32_t>(DEVMLogLevelType::DEVM_INFO);
};

#define DEVM_LOG_HEAD          \
    " pid:" << getpid() << " " \
            << "tid:" << (long int)syscall(__NR_gettid) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define DEVM_LOG_CRITICAL hozon::netaos::devm_server::DevmServerLogger::GetInstance().GetLogger()->LogCritical() << DEVM_LOG_HEAD
#define DEVM_LOG_ERROR hozon::netaos::devm_server::DevmServerLogger::GetInstance().GetLogger()->LogError() << DEVM_LOG_HEAD
#define DEVM_LOG_WARN hozon::netaos::devm_server::DevmServerLogger::GetInstance().GetLogger()->LogWarn() << DEVM_LOG_HEAD
#define DEVM_LOG_INFO hozon::netaos::devm_server::DevmServerLogger::GetInstance().GetLogger()->LogInfo() << DEVM_LOG_HEAD
#define DEVM_LOG_DEBUG hozon::netaos::devm_server::DevmServerLogger::GetInstance().GetLogger()->LogDebug() << DEVM_LOG_HEAD
#define DEVM_LOG_TRACE hozon::netaos::devm_server::DevmServerLogger::GetInstance().GetLogger()->LogTrace() << DEVM_LOG_HEAD

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
#endif  // end of DEVM_LOGGER_H_

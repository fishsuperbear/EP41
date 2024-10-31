/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */

#ifndef DEVM_CLIENT_LOGGER_H_
#define DEVM_CLIENT_LOGGER_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <memory>
#include <string>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace devm {
/* devm_server log class */
class DevmClientLogger {
   public:
    static DevmClientLogger& GetInstance() {
        static DevmClientLogger instance;
        return instance;
    }
    ~DevmClientLogger() {}
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
    DevmClientLogger() {}
    std::shared_ptr<hozon::netaos::log::Logger> logger_{hozon::netaos::log::CreateLogger("DEVMC", "NETAOS DEVM CLIENT", hozon::netaos::log::LogLevel::kInfo)};
    int32_t level_ = static_cast<int32_t>(DEVMLogLevelType::DEVM_INFO);
};

#define DEVM_LOG_HEAD          \
    " pid:" << getpid() << " " \
            << "tid:" << (long int)syscall(__NR_gettid) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define DEVM_LOG_CRITICAL hozon::netaos::devm::DevmClientLogger::GetInstance().GetLogger()->LogCritical() << DEVM_LOG_HEAD
#define DEVM_LOG_ERROR hozon::netaos::devm::DevmClientLogger::GetInstance().GetLogger()->LogError() << DEVM_LOG_HEAD
#define DEVM_LOG_WARN hozon::netaos::devm::DevmClientLogger::GetInstance().GetLogger()->LogWarn() << DEVM_LOG_HEAD
#define DEVM_LOG_INFO hozon::netaos::devm::DevmClientLogger::GetInstance().GetLogger()->LogInfo() << DEVM_LOG_HEAD
#define DEVM_LOG_DEBUG hozon::netaos::devm::DevmClientLogger::GetInstance().GetLogger()->LogDebug() << DEVM_LOG_HEAD
#define DEVM_LOG_TRACE hozon::netaos::devm::DevmClientLogger::GetInstance().GetLogger()->LogTrace() << DEVM_LOG_HEAD

}  // namespace devm
}  // namespace netaos
}  // namespace hozon
#endif  // end of DEVM_LOGGER_H_

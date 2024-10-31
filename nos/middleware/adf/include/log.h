#pragma once

#include <string.h>
#include <sys/syscall.h>
#include <iostream>
#include <thread>
#include "log/include/logging.h"

using namespace hozon::netaos::log;

namespace hozon {
namespace netaos {
namespace adf {
class NodeLogger {
   public:
    static NodeLogger& GetInstance() {
        static NodeLogger instance;
        return instance;
    }

    ~NodeLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

    void CreateLogger(std::string ctx_id, std::string ctx_description, hozon::netaos::log::LogLevel log_level) {
        auto log_{hozon::netaos::log::CreateLogger(ctx_id, ctx_description, log_level)};
        logger_ = log_;
    }

   private:
    NodeLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};

#define NODE_LOG_HEAD __FUNCTION__ << "() " << strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "

#define NODE_LOGGER hozon::netaos::adf::NodeLogger::GetInstance().GetLogger()
#define APP_OP_LOGGER \
    hozon::netaos::log::CreateOperationLogger("APP", "app description", hozon::netaos::log::LogLevel::kInfo)

#define NODE_LOG_CRITICAL                                                                                   \
    (!NODE_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kCritical)) ? (NODE_LOGGER->LogCritical() << "") \
                                                                       : (NODE_LOGGER->LogCritical() << NODE_LOG_HEAD)
#define NODE_LOG_ERROR                                                                                \
    (!NODE_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kError)) ? (NODE_LOGGER->LogError() << "") \
                                                                    : (NODE_LOGGER->LogError() << NODE_LOG_HEAD)
#define NODE_LOG_WARN                                                                               \
    (!NODE_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kWarn)) ? (NODE_LOGGER->LogWarn() << "") \
                                                                   : (NODE_LOGGER->LogWarn() << NODE_LOG_HEAD)
#define NODE_LOG_INFO                                                                               \
    (!NODE_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kInfo)) ? (NODE_LOGGER->LogInfo() << "") \
                                                                   : (NODE_LOGGER->LogInfo() << NODE_LOG_HEAD)
#define NODE_LOG_DEBUG                                                                                \
    (!NODE_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kDebug)) ? (NODE_LOGGER->LogDebug() << "") \
                                                                    : (NODE_LOGGER->LogDebug() << NODE_LOG_HEAD)
#define NODE_LOG_TRACE                                                                                \
    (!NODE_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kTrace)) ? (NODE_LOGGER->LogTrace() << "") \
                                                                    : (NODE_LOGGER->LogTrace() << NODE_LOG_HEAD)

#define APP_OP_LOG_CRITICAL                                              \
    (!APP_OP_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kCritical)) \
        ? (APP_OP_LOGGER->LogCritical() << "")                           \
        : (APP_OP_LOGGER->LogCritical() << NODE_LOG_HEAD)
#define APP_OP_LOG_ERROR                                                                                  \
    (!APP_OP_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kError)) ? (APP_OP_LOGGER->LogError() << "") \
                                                                      : (APP_OP_LOGGER->LogError() << NODE_LOG_HEAD)
#define APP_OP_LOG_WARN                                                                                 \
    (!APP_OP_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kWarn)) ? (APP_OP_LOGGER->LogWarn() << "") \
                                                                     : (APP_OP_LOGGER->LogWarn() << NODE_LOG_HEAD)
#define APP_OP_LOG_INFO                                                                                 \
    (!APP_OP_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kInfo)) ? (APP_OP_LOGGER->LogInfo() << "") \
                                                                     : (APP_OP_LOGGER->LogInfo() << NODE_LOG_HEAD)
#define APP_OP_LOG_DEBUG                                                                                  \
    (!APP_OP_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kDebug)) ? (APP_OP_LOGGER->LogDebug() << "") \
                                                                      : (APP_OP_LOGGER->LogDebug() << NODE_LOG_HEAD)
#define APP_OP_LOG_TRACE                                                                                  \
    (!APP_OP_LOGGER->IsEnabled(hozon::netaos::log::LogLevel::kTrace)) ? (APP_OP_LOGGER->LogTrace() << "") \
                                                                      : (APP_OP_LOGGER->LogTrace() << NODE_LOG_HEAD)

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
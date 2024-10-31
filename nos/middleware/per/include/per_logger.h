/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: log
 * Created on: Feb 7, 2023
 *
 */

#ifndef MIDDLEWARE_PER_INCLUDE_PER_LOGGER_H_
#define MIDDLEWARE_PER_INCLUDE_PER_LOGGER_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <memory>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace per {
/* CONFIG log class */
class PerLogger {
 public:
    static PerLogger& GetInstance() {
        static PerLogger instance;
        return instance;
    }
    ~PerLogger() {}

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

 private:
    PerLogger() {}
    std::shared_ptr<hozon::netaos::log::Logger> logger_{hozon::netaos::log::CreateLogger("PER", "NETAOS PER", hozon::netaos::log::LogLevel::kInfo)};
};

#define PER_LOG_HEAD           \
    " pid:" << getpid() << " " \
            << "tid:" << (int64_t)syscall(__NR_gettid) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define PER_LOG_CRITICAL hozon::netaos::per::PerLogger::GetInstance().GetLogger()->LogCritical() << PER_LOG_HEAD
#define PER_LOG_ERROR hozon::netaos::per::PerLogger::GetInstance().GetLogger()->LogError() << PER_LOG_HEAD
#define PER_LOG_WARN hozon::netaos::per::PerLogger::GetInstance().GetLogger()->LogWarn() << PER_LOG_HEAD
#define PER_LOG_INFO hozon::netaos::per::PerLogger::GetInstance().GetLogger()->LogInfo() << PER_LOG_HEAD
#define PER_LOG_DEBUG hozon::netaos::per::PerLogger::GetInstance().GetLogger()->LogDebug() << PER_LOG_HEAD
#define PER_LOG_TRACE hozon::netaos::per::PerLogger::GetInstance().GetLogger()->LogTrace() << PER_LOG_HEAD

}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_PER_LOGGER_H_

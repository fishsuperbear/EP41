#pragma once

#include <unistd.h>
#include <iostream>
#include <memory>
#include "log/include/logger.h"
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace hz_time {
class TimeLogger {
   private:
    /* data */
    TimeLogger() {}

    std::shared_ptr<hozon::netaos::log::Logger> logger_;

   public:
    static TimeLogger& GetInstance() {
        static TimeLogger instance;
        return instance;
    }

    ~TimeLogger() {}

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

    void InitLogger(log::LogLevel log_level, uint32_t log_mode, std::string log_path) { hozon::netaos::log::InitLogging("TIME", "TIME", log_level, log_mode, log_path, 10, 20); }

    void CreateLogger(log::LogLevel log_level) {
        auto log_{hozon::netaos::log::CreateLogger("TIME", "TIME", log_level)};
        logger_ = log_;
    }
};

#define TIME_LOG_HEAD "@" << getpid() << " " << strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "

#define TIME_LOG_CRITICAL (TimeLogger::GetInstance().GetLogger()->LogCrirical() << TIME_LOG_HEAD)
#define TIME_LOG_ERROR (TimeLogger::GetInstance().GetLogger()->LogError() << TIME_LOG_HEAD)
#define TIME_LOG_WARN (TimeLogger::GetInstance().GetLogger()->LogWarn() << TIME_LOG_HEAD)
#define TIME_LOG_INFO (TimeLogger::GetInstance().GetLogger()->LogInfo() << TIME_LOG_HEAD)
#define TIME_LOG_DEBUG (TimeLogger::GetInstance().GetLogger()->LogDebug() << TIME_LOG_HEAD)
#define TIME_LOG_TRACE (TimeLogger::GetInstance().GetLogger()->LogTrace() << TIME_LOG_HEAD)

class TimeEarlyLogger {
   public:
    ~TimeEarlyLogger() { std::cout << std::endl; }

    template <typename T>
    TimeEarlyLogger& operator<<(const T& value) {
        std::cout << value;
        return *this;
    }
};

#define TIME_EARLY_LOG TimeEarlyLogger() << TIME_LOG_HEAD
}  // namespace hz_time
}  // namespace netaos
}  // namespace hozon
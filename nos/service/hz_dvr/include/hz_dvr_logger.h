#pragma once

#include <iostream>
#include <memory>
#include "log/include/logger.h"
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace hz_dvr {
class DvrLogger {
   private:
    /* data */
    DvrLogger() {}

    std::shared_ptr<hozon::netaos::log::Logger> logger_;

   public:
    static DvrLogger& GetInstance() {
        static DvrLogger instance;
        return instance;
    }

    ~DvrLogger() {}

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

    void InitLogger(log::LogLevel log_level, uint32_t log_mode, std::string log_path) {
        hozon::netaos::log::InitLogging("DVR", "DVR", log_level, log_mode, log_path, 10, 20);
    }

    void CreateLogger(log::LogLevel log_level) {
        auto log_{hozon::netaos::log::CreateLogger("DVR", "DVR", log_level)};
        logger_ = log_;
    }
};

#define DVR_LOG_HEAD "@" << getpid() << " " << strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "

#define DVR_LOG_CRITICAL (DvrLogger::GetInstance().GetLogger()->LogCrirical() << DVR_LOG_HEAD)
#define DVR_LOG_ERROR (DvrLogger::GetInstance().GetLogger()->LogError() << DVR_LOG_HEAD)
#define DVR_LOG_WARN (DvrLogger::GetInstance().GetLogger()->LogWarn() << DVR_LOG_HEAD)
#define DVR_LOG_INFO (DvrLogger::GetInstance().GetLogger()->LogInfo() << DVR_LOG_HEAD)
#define DVR_LOG_DEBUG (DvrLogger::GetInstance().GetLogger()->LogDebug() << DVR_LOG_HEAD)
#define DVR_LOG_TRACE (DvrLogger::GetInstance().GetLogger()->LogTrace() << DVR_LOG_HEAD)

class DvrEarlyLogger {
   public:
    ~DvrEarlyLogger() { std::cout << std::endl; }

    template <typename T>
    DvrEarlyLogger& operator<<(const T& value) {
        std::cout << value;
        return *this;
    }
};

#define DVR_EARLY_LOG DvrEarlyLogger() << DVR_LOG_HEAD

}  // namespace hz_dvr
}  // namespace netaos
}  // namespace hozon
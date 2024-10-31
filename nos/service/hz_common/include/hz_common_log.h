#pragma once

#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "log/include/logging.h"

namespace hozon {
namespace neatos {
namespace common {

using namespace hozon::netaos::log;

class PlatformCommonLogger {
   public:
    virtual ~PlatformCommonLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

    static PlatformCommonLogger& GetInstance() {
        static PlatformCommonLogger instance;
        return instance;
    }

    void InitLogger(std::string log_name,
                    // std::string log_description,
                    LogLevel log_level, uint32_t log_mode, std::string log_path) {
        hozon::netaos::log::InitLogging(log_name, "Common Logger",  static_cast<LogLevel>(log_level), log_mode, log_path, 10, 20);
    }

    void CreateLogger(std::string ctx_id, std::string ctx_description, LogLevel log_level) {
        auto log_{hozon::netaos::log::CreateLogger(ctx_id, ctx_description, log_level)};
        logger_ = log_;
    }
   private:

    PlatformCommonLogger() {}
    PlatformCommonLogger& operator=(const PlatformCommonLogger&);
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};

#define PLATFORM_COMMON_LOG_HEAD getpid() << " " << (long int)syscall(__NR_gettid) << " " << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << "(" << __LINE__ << ") | "

#define LOG_CRITICIAL hozon::netaos::common::PlatformCommonLogger::GetInstance().GetLogger()->LogCritical() << PLATFORM_COMMON_LOG_HEAD
#define LOG_ERROR hozon::netaos::common::PlatformCommonLogger::GetInstance().GetLogger()->LogError() << PLATFORM_COMMON_LOG_HEAD
#define LOG_WARN hozon::netaos::common::PlatformCommonLogger::GetInstance().GetLogger()->LogWarn() << PLATFORM_COMMON_LOG_HEAD
#define LOG_INFO hozon::netaos::common::PlatformCommonLogger::GetInstance().GetLogger()->LogInfo() << PLATFORM_COMMON_LOG_HEAD
#define LOG_DEBUG hozon::netaos::common::PlatformCommonLogger::GetInstance().GetLogger()->LogDebug() << PLATFORM_COMMON_LOG_HEAD
#define LOG_TRACE hozon::netaos::common::PlatformCommonLogger::GetInstance().GetLogger()->LogTrace() << PLATFORM_COMMON_LOG_HEAD

}  // namespace common
}  // namespace neatos
}  // namespace hozon

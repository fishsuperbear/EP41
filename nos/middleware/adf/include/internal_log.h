#pragma once

#include <stdarg.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "log/include/logging.h"

using namespace hozon::netaos::log;

namespace hozon {
namespace netaos {
namespace adf {
class AdfLogger {
   public:
    static AdfLogger& GetInstance() {
        static AdfLogger instance;
        return instance;
    }

    ~AdfLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

    void InitLogger(std::string log_name, std::string log_description, LogLevel log_level, uint32_t log_mode,
                    std::string log_path) {
        hozon::netaos::log::InitLogging(log_name, log_description, log_level, log_mode, log_path, 10, 20, true);
    }

    void CreateLogger(std::string ctx_id, std::string ctx_description, LogLevel log_level) {
        auto log_{hozon::netaos::log::CreateLogger(ctx_id, ctx_description, log_level)};
        logger_ = log_;
    }

   private:
    AdfLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};

#define ADF_LOG_HEAD strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "

#define ADF_LOG_CRITICAL (AdfLogger::GetInstance().GetLogger()->LogCritical() << ADF_LOG_HEAD)
#define ADF_LOG_ERROR (AdfLogger::GetInstance().GetLogger()->LogError() << ADF_LOG_HEAD)
#define ADF_LOG_WARN (AdfLogger::GetInstance().GetLogger()->LogWarn() << ADF_LOG_HEAD)
#define ADF_LOG_INFO (AdfLogger::GetInstance().GetLogger()->LogInfo() << ADF_LOG_HEAD)
#define ADF_LOG_DEBUG (AdfLogger::GetInstance().GetLogger()->LogDebug() << ADF_LOG_HEAD)
#define ADF_LOG_TRACE (AdfLogger::GetInstance().GetLogger()->LogTrace() << ADF_LOG_HEAD)

class AdfEarlyLogger {
   public:
    ~AdfEarlyLogger() { std::cout << std::endl; }

    template <typename T>
    AdfEarlyLogger& operator<<(const T& value) {
        std::cout << value;
        return *this;
    }
};

#define ADF_EARLY_LOG AdfEarlyLogger() << ADF_LOG_HEAD
}  // namespace adf
}  // namespace netaos
}  // namespace hozon
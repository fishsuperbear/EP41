#pragma once

#include "log/include/logging.h"
#include <unistd.h>

using namespace hozon::netaos::log;

class LidarLogger {
public:
    static LidarLogger& GetInstance() {
        static LidarLogger instance;
        return instance;
    }
    ~LidarLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

    void InitLogger(uint8_t logLevel,uint32_t logMode) {
        hozon::netaos::log::InitLogging(
            "lidar",
            "lidar logger instance for test",
            // LogLevel::kInfo,
            // LogLevel::kTrace,
            static_cast<hozon::netaos::log::LogLevel>(logLevel),
            // HZ_LOG2FILE,
            // HZ_LOG2CONSOLE | HZ_LOG2FILE,
            logMode,
            "/opt/usr/log/soc_log",
            10,
            20
        );

        logger_ = hozon::netaos::log::CreateLogger("lidar", "lidar logger instance",
                                                    static_cast<hozon::netaos::log::LogLevel>(logLevel));
    }

private:
    LidarLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};

#define LIDAR_LOG_HEAD  "pid:"<< getpid() << " " << strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "

#define LIDAR_LOG_CRITICAL             (LidarLogger::GetInstance().GetLogger()->LogCritical()<<LIDAR_LOG_HEAD)
#define LIDAR_LOG_ERROR                (LidarLogger::GetInstance().GetLogger()->LogError() <<LIDAR_LOG_HEAD)
#define LIDAR_LOG_WARN                 (LidarLogger::GetInstance().GetLogger()->LogWarn() <<LIDAR_LOG_HEAD)
#define LIDAR_LOG_INFO                 (LidarLogger::GetInstance().GetLogger()->LogInfo() << LIDAR_LOG_HEAD)
#define LIDAR_LOG_DEBUG                (LidarLogger::GetInstance().GetLogger()->LogDebug() <<LIDAR_LOG_HEAD)
#define LIDAR_LOG_TRACE                (LidarLogger::GetInstance().GetLogger()->LogTrace() <<LIDAR_LOG_HEAD)

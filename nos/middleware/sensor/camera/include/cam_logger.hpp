#pragma once

#include <iostream>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace camera {

class CAMLogger
{
public:
    static CAMLogger& GetInstance() {
        static CAMLogger instance;
        return instance;
    }
    ~CAMLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger>  GetLogger() const { return logger_; }

private:
    CAMLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_ { hozon::netaos::log::CreateLogger("Cam", "NETAOS CAM",
                        hozon::netaos::log::LogLevel::kInfo) };
};

#define CAM_LOG_HEAD  \
    "["<< "Sensor ID: "<< _sensor_id << "] "

#define CAM_LOG_CRITICAL_WITH_HEAD       CAM_LOG_CRITICAL << CAM_LOG_HEAD
#define CAM_LOG_ERROR_WITH_HEAD          CAM_LOG_ERROR << CAM_LOG_HEAD
#define CAM_LOG_WARN_WITH_HEAD           CAM_LOG_WARN << CAM_LOG_HEAD
#define CAM_LOG_INFO_WITH_HEAD           CAM_LOG_INFO << CAM_LOG_HEAD
#define CAM_LOG_DEBUG_WITH_HEAD          CAM_LOG_DEBUG << CAM_LOG_HEAD
#define CAM_LOG_TRACE_WITH_HEAD          CAM_LOG_TRACE << CAM_LOG_HEAD

#define CAM_LOG_CRITICAL         (CAMLogger::GetInstance().GetLogger()->LogCritical())
#define CAM_LOG_ERROR            (CAMLogger::GetInstance().GetLogger()->LogError())
#define CAM_LOG_WARN             (CAMLogger::GetInstance().GetLogger()->LogWarn())
#define CAM_LOG_INFO             (CAMLogger::GetInstance().GetLogger()->LogInfo())
#define CAM_LOG_DEBUG            (CAMLogger::GetInstance().GetLogger()->LogDebug())
#define CAM_LOG_TRACE            (CAMLogger::GetInstance().GetLogger()->LogTrace())

}
}
}

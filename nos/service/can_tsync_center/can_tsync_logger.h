#pragma once

#include <iostream>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace tsync {

class CANTSyncLogger
{
public:
    static CANTSyncLogger& GetInstance() {
        static CANTSyncLogger instance;
        return instance;
    }
    ~CANTSyncLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger>  GetLogger() const { return logger_; }

private:
    CANTSyncLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_ { hozon::netaos::log::CreateLogger("CTSC", "CAN TSync",
                        hozon::netaos::log::LogLevel::kInfo) };
};

#define CTSC_HEAD                        _config.interface << " | "

#define CTSC_LOG_CRITICAL               (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogCritical())
#define CTSC_LOG_ERROR                  (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogError())
#define CTSC_LOG_WARN                   (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogWarn())
#define CTSC_LOG_INFO                   (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogInfo())
#define CTSC_LOG_DEBUG                  (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogDebug())
#define CTSC_LOG_TRACE                  (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogTrace())

#define CTSC_LOG_CRITICAL_HEAD          (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogCritical() << CTSC_HEAD)
#define CTSC_LOG_ERROR_HEAD             (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogError() << CTSC_HEAD)
#define CTSC_LOG_WARN_HEAD              (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogWarn() << CTSC_HEAD)
#define CTSC_LOG_INFO_HEAD              (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogInfo() << CTSC_HEAD)
#define CTSC_LOG_DEBUG_HEAD             (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogDebug() << CTSC_HEAD)
#define CTSC_LOG_TRACE_HEAD             (hozon::netaos::tsync::CANTSyncLogger::GetInstance().GetLogger()->LogTrace() << CTSC_HEAD)

}
}
}
#pragma once

#include <iostream>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace em {

class LBLogger
{
public:
    static LBLogger& GetInstance() {
        static LBLogger instance;
        return instance;
    }
    ~LBLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger>  GetLogger() const { return logger_; }

private:
    LBLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_ { hozon::netaos::log::CreateLogger("EM", "NETAOS EM LIB",
                        hozon::netaos::log::LogLevel::kInfo) };
};

#define LB_LOG_HEAD ""
#define LB_LOG_CRITICAL_WITH_HEAD      LB_LOG_CRITICAL <<LB_LOG_HEAD
#define LB_LOG_ERROR_WITH_HEAD         LB_LOG_ERROR << LB_LOG_HEAD
#define LB_LOG_WARN_WITH_HEAD           LB_LOG_WARN << LB_LOG_HEAD
#define LB_LOG_INFO_WITH_HEAD           LB_LOG_INFO << LB_LOG_HEAD
#define LB_LOG_DEBUG_WITH_HEAD          LB_LOG_DEBUG << LB_LOG_HEAD
#define LB_LOG_TRACE_WITH_HEAD          LB_LOG_TRACE << LB_LOG_HEAD

#define LB_LOG_CRITICAL         (LBLogger::GetInstance().GetLogger()->LogCritical())
#define LB_LOG_ERROR            (LBLogger::GetInstance().GetLogger()->LogError())
#define LB_LOG_WARN             (LBLogger::GetInstance().GetLogger()->LogWarn())
#define LB_LOG_INFO             (LBLogger::GetInstance().GetLogger()->LogInfo())
#define LB_LOG_DEBUG            (LBLogger::GetInstance().GetLogger()->LogDebug())
#define LB_LOG_TRACE            (LBLogger::GetInstance().GetLogger()->LogTrace())

#define LB_CONFIG_LOG_HEAD getpid() << " " << (long int)syscall(186) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define LB_LOG_CRITICAL_WITH_HEAD_FUNC LB_LOG_CRITICAL << LB_CONFIG_LOG_HEAD
#define LB_LOG_ERROR_WITH_HEAD_FUNC LB_LOG_ERROR << LB_CONFIG_LOG_HEAD
#define LB_LOG_WARN_WITH_HEAD_FUNC LB_LOG_WARN << LB_CONFIG_LOG_HEAD
#define LB_LOG_INFO_WITH_HEAD_FUNC LB_LOG_INFO << LB_CONFIG_LOG_HEAD
#define LB_LOG_DEBUG_WITH_HEAD_FUNC LB_LOG_DEBUG << LB_CONFIG_LOG_HEAD
#define LB_LOG_TRACE_WITH_HEAD_FUNC LB_LOG_TRACE << LB_CONFIG_LOG_HEAD

}
}
}

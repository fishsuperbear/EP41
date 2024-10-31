#pragma once

#include <iostream>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace cm {

class CMLogger
{
public:
    static CMLogger& GetInstance() {
        static CMLogger instance;
        return instance;
    }
    ~CMLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger>  GetLogger() const { return logger_; }

private:
    CMLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_ { hozon::netaos::log::CreateLogger("CM", "NETAOS DDS",
                        hozon::netaos::log::LogLevel::kInfo) };
};

#define CM_LOG_HEAD  \
    "["<< "domain: "<< _domain << " topic: " << _topic << "] "

#define CM_LOG_CRITICAL_WITH_HEAD       CM_LOG_CRITICAL << CM_LOG_HEAD
#define CM_LOG_ERROR_WITH_HEAD          CM_LOG_ERROR << CM_LOG_HEAD
#define CM_LOG_WARN_WITH_HEAD           CM_LOG_WARN << CM_LOG_HEAD
#define CM_LOG_INFO_WITH_HEAD           CM_LOG_INFO << CM_LOG_HEAD
#define CM_LOG_DEBUG_WITH_HEAD          CM_LOG_DEBUG << CM_LOG_HEAD
#define CM_LOG_TRACE_WITH_HEAD          CM_LOG_TRACE << CM_LOG_HEAD

#define CM_LOG_CRITICAL         (CMLogger::GetInstance().GetLogger()->LogCritical())
#define CM_LOG_ERROR            (CMLogger::GetInstance().GetLogger()->LogError())
#define CM_LOG_WARN             (CMLogger::GetInstance().GetLogger()->LogWarn())
#define CM_LOG_INFO             (CMLogger::GetInstance().GetLogger()->LogInfo())
#define CM_LOG_DEBUG            (CMLogger::GetInstance().GetLogger()->LogDebug())
#define CM_LOG_TRACE            (CMLogger::GetInstance().GetLogger()->LogTrace())

#define CONFIG_LOG_HEAD getpid() << " " << (long int)syscall(186) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define CM_LOG_CRITICAL_WITH_HEAD_FUNC CM_LOG_CRITICAL << CONFIG_LOG_HEAD
#define CM_LOG_ERROR_WITH_HEAD_FUNC CM_LOG_ERROR << CONFIG_LOG_HEAD
#define CM_LOG_WARN_WITH_HEAD_FUNC CM_LOG_WARN << CONFIG_LOG_HEAD
#define CM_LOG_INFO_WITH_HEAD_FUNC CM_LOG_INFO << CONFIG_LOG_HEAD
#define CM_LOG_DEBUG_WITH_HEAD_FUNC CM_LOG_DEBUG << CONFIG_LOG_HEAD
#define CM_LOG_TRACE_WITH_HEAD_FUNC CM_LOG_TRACE << CONFIG_LOG_HEAD

}
}
}

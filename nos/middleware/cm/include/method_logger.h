#pragma once

#include <iostream>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace cm {

class MDLogger
{
public:
    static MDLogger& GetInstance() {
        static MDLogger instance;
        return instance;
    }
    ~MDLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger>  GetLogger() const { return logger_; }

private:
    MDLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_ { hozon::netaos::log::CreateLogger("MD", "NETAOS DDS",
                        hozon::netaos::log::LogLevel::kInfo) };
};

#define MD_LOG_HEAD  \
    "["<< "domain: "<< _domain << " topic: " << _topic << "] "

#define MD_LOG_CRITICAL_WITH_HEAD       MD_LOG_CRITICAL << MD_LOG_HEAD
#define MD_LOG_ERROR_WITH_HEAD          MD_LOG_ERROR << MD_LOG_HEAD
#define MD_LOG_WARN_WITH_HEAD           MD_LOG_WARN << MD_LOG_HEAD
#define MD_LOG_INFO_WITH_HEAD           MD_LOG_INFO << MD_LOG_HEAD
#define MD_LOG_DEBUG_WITH_HEAD          MD_LOG_DEBUG << MD_LOG_HEAD
#define MD_LOG_TRACE_WITH_HEAD          MD_LOG_TRACE << MD_LOG_HEAD

#define MD_LOG_CRITICAL         (MDLogger::GetInstance().GetLogger()->LogCritical())
#define MD_LOG_ERROR            (MDLogger::GetInstance().GetLogger()->LogError())
#define MD_LOG_WARN             (MDLogger::GetInstance().GetLogger()->LogWarn())
#define MD_LOG_INFO             (MDLogger::GetInstance().GetLogger()->LogInfo())
#define MD_LOG_DEBUG            (MDLogger::GetInstance().GetLogger()->LogDebug())
#define MD_LOG_TRACE            (MDLogger::GetInstance().GetLogger()->LogTrace())

#define CONFIG_LOG_HEAD getpid() << " " << (long int)syscall(186) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define MD_LOG_CRITICAL_WITH_HEAD_FUNC MD_LOG_CRITICAL << CONFIG_LOG_HEAD
#define MD_LOG_ERROR_WITH_HEAD_FUNC MD_LOG_ERROR << CONFIG_LOG_HEAD
#define MD_LOG_WARN_WITH_HEAD_FUNC MD_LOG_WARN << CONFIG_LOG_HEAD
#define MD_LOG_INFO_WITH_HEAD_FUNC MD_LOG_INFO << CONFIG_LOG_HEAD
#define MD_LOG_DEBUG_WITH_HEAD_FUNC MD_LOG_DEBUG << CONFIG_LOG_HEAD
#define MD_LOG_TRACE_WITH_HEAD_FUNC MD_LOG_TRACE << CONFIG_LOG_HEAD

}
}
}

#pragma once

#include "log/include/logging.h"
#include "nvscierror.h"
#include <string.h>

namespace hozon {
namespace netaos {
namespace nv {

class NVSLogger {
public:
    static NVSLogger& GetInstance() {
        static NVSLogger instance;
        return instance;
    }
    ~NVSLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

private:
    NVSLogger(){};
    std::shared_ptr<hozon::netaos::log::Logger> logger_ { hozon::netaos::log::CreateLogger("NVS", "NvStream adapter",
                        hozon::netaos::log::LogLevel::kInfo) };
};

constexpr log::LogHex32 LogHexNvErr(NvSciError err) {
    return log::loghex((uint32_t)err);
}

}
}
}

#define NVS_LOG_HEAD                    strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "

#define NVS_LOG_CRITICAL                (NVSLogger::GetInstance().GetLogger()->LogCritical()) << NVS_LOG_HEAD
#define NVS_LOG_ERROR                   (NVSLogger::GetInstance().GetLogger()->LogError()) << NVS_LOG_HEAD
#define NVS_LOG_WARN                    (NVSLogger::GetInstance().GetLogger()->LogWarn()) << NVS_LOG_HEAD
#define NVS_LOG_INFO                    (NVSLogger::GetInstance().GetLogger()->LogInfo()) << NVS_LOG_HEAD
#define NVS_LOG_DEBUG                   (NVSLogger::GetInstance().GetLogger()->LogDebug()) << NVS_LOG_HEAD
#define NVS_LOG_TRACE                   (NVSLogger::GetInstance().GetLogger()->LogTrace()) << NVS_LOG_HEAD
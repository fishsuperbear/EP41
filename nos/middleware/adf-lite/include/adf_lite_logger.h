#pragma once

#include <iostream>
#include <unordered_map>
#include "log/include/logging.h"
#include <string.h>

namespace hozon {
namespace netaos {
namespace adf_lite {

enum LogLevel : int32_t {
    ADF_LOG_LEVEL_VERBOSE = 0,
    ADF_LOG_LEVEL_DEBUG = 1,
    ADF_LOG_LEVEL_INFO = 2,
    ADF_LOG_LEVEL_WARN = 3,
    ADF_LOG_LEVEL_ERROR = 4,
    ADF_LOG_LEVEL_FATAL = 5,
    ADF_LOG_LEVEL_OFF = 6,
};

enum LogMode : int32_t {
    ADF_LOG_MODE_REMOTE = 0,
    ADF_LOG_MODE_CONSOLE = 1,
    ADF_LOG_MODE_FILE = 2,
    ADF_LOG_MODE_REMOTE_CONSOLE = 3,
    ADF_LOG_MODE_REMOTE_FILE = 4,
    ADF_LOG_MODE_CONSOLE_FILE = 5,
    ADF_LOG_MODE_REMOTE_CONSOLE_FILE = 6,
};

class CtxLogger {
public:
    CtxLogger() {}
    ~CtxLogger() {}

    void Init(const std::string& ctx_id, const LogLevel level) {
        m_logger = hozon::netaos::log::CreateLogger(ctx_id, ctx_id, ToPlatformLogLevel(level));
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() {
        return m_logger;
    }

    static inline hozon::netaos::log::LogLevel ToPlatformLogLevel(LogLevel level) {
        return static_cast<hozon::netaos::log::LogLevel>(level);
    }

private:
    std::shared_ptr<hozon::netaos::log::Logger> m_logger;
};

#define CTX_LOG_HEAD                    strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "
#define CTX_LOG_ENABLE(ctx, level)      true
#define CTX_FATAL_OSTREAM(ctx)          ctx.GetLogger()->LogCritical()
#define CTX_ERROR_OSTREAM(ctx)          ctx.GetLogger()->LogError()
#define CTX_WARN_OSTREAM(ctx)           ctx.GetLogger()->LogWarn()
#define CTX_INFO_OSTREAM(ctx)           ctx.GetLogger()->LogInfo()
#define CTX_DEBUG_OSTREAM(ctx)          ctx.GetLogger()->LogDebug()
#define CTX_VERBOSE_OSTREAM(ctx)        ctx.GetLogger()->LogTrace()

#define CTX_LOG_FATAL(ctx) \
    (!CTX_LOG_ENABLE(ctx, ara::log::LogLevel::kFatal)) ? (CTX_FATAL_OSTREAM(ctx) << "") : (CTX_FATAL_OSTREAM(ctx)) << CTX_LOG_HEAD
#define CTX_LOG_ERROR(ctx) \
    (!CTX_LOG_ENABLE(ctx, ara::log::LogLevel::kError)) ? (CTX_ERROR_OSTREAM(ctx) << "") : (CTX_ERROR_OSTREAM(ctx)) << CTX_LOG_HEAD
#define CTX_LOG_WARN(ctx) \
    (!CTX_LOG_ENABLE(ctx, ara::log::LogLevel::kWarn)) ? (CTX_WARN_OSTREAM(ctx) << "") : (CTX_WARN_OSTREAM(ctx)) << CTX_LOG_HEAD
#define CTX_LOG_INFO(ctx) \
    (!CTX_LOG_ENABLE(ctx, ara::log::LogLevel::kInfo)) ? (CTX_INFO_OSTREAM(ctx) << "") : (CTX_INFO_OSTREAM(ctx)) << CTX_LOG_HEAD
#define CTX_LOG_DEBUG(ctx) \
    (!CTX_LOG_ENABLE(ctx, ara::log::LogLevel::kDebug)) ? (CTX_DEBUG_OSTREAM(ctx) << "") : (CTX_DEBUG_OSTREAM(ctx)) << CTX_LOG_HEAD
#define CTX_LOG_VERBOSE(ctx) \
    (!CTX_LOG_ENABLE(ctx, ara::log::LogLevel::kVerbose)) ? (CTX_VERBOSE_OSTREAM(ctx) << "") : (CTX_VERBOSE_OSTREAM(ctx)) << CTX_LOG_HEAD

}
}
}
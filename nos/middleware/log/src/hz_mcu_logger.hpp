#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <mutex>

#include "logger.h"

namespace hozon {
namespace netaos {
namespace log {


class HzMcuLogger final: public Logger {
 public:
    HzMcuLogger(std::string appId, std::string ctxId, LogLevel ctxDefLogLevel);

    ~HzMcuLogger();

    virtual LogStream LogCritical() noexcept;

    virtual LogStream LogError() noexcept;

    virtual LogStream LogWarn() noexcept;

    virtual LogStream LogInfo() noexcept;
 
    virtual LogStream LogDebug() noexcept;

    virtual LogStream LogTrace() noexcept;

    virtual bool IsOperationLog() noexcept;

    virtual bool IsEnabled(LogLevel level) noexcept;
    
    virtual bool SetLogLevel(const LogLevel level) noexcept;

    void LogOut(LogLevel level, const std::string& message);

    LogLevel GetOutputLogLevel() const noexcept;
    void ForceSetCtxLogLevel(const LogLevel level);

    std::string GetCtxId() const noexcept;
    std::string GetAppId() const noexcept;

 private:
   std::string ctxID_;
   std::string appID_;
   LogLevel level_;
};


}
}
}


#include <iostream>
#include "hz_mcu_logger.hpp"
#include "logstream.h"
#include "log_manager.hpp"

namespace hozon {
namespace netaos {
namespace log {

HzMcuLogger::HzMcuLogger(std::string appId, std::string ctxId, LogLevel ctxDefLogLevel)
:Logger(),
appID_(appId),
ctxID_(ctxId),
level_(ctxDefLogLevel) {}

HzMcuLogger::~HzMcuLogger()
{
}

LogStream HzMcuLogger::LogCritical() noexcept
{
    return LogStream{LogLevel::kCritical, this};
}

LogStream HzMcuLogger::LogError() noexcept
{
    return LogStream{LogLevel::kError, this};
}

LogStream HzMcuLogger::LogWarn() noexcept
{
    return LogStream{LogLevel::kWarn, this};
}

LogStream HzMcuLogger::LogInfo() noexcept
{
    return LogStream{LogLevel::kInfo, this};
}

LogStream HzMcuLogger::LogDebug() noexcept
{
    return LogStream{LogLevel::kDebug, this};
}

LogStream HzMcuLogger::LogTrace() noexcept
{
    return LogStream{LogLevel::kTrace, this};
}

bool HzMcuLogger::IsOperationLog() noexcept
{
    return false;
}

bool HzMcuLogger::IsEnabled(LogLevel level) noexcept
{
    return true;
}
    
bool HzMcuLogger::SetLogLevel(const LogLevel level) noexcept
{
    return true;
}

void HzMcuLogger::LogOut(LogLevel level, const std::string& message)
{
    HzLogManager::GetInstance()->mcuLogout(appID_, level, message);
}

LogLevel HzMcuLogger::GetOutputLogLevel() const noexcept
{
    return level_;
}

void HzMcuLogger::ForceSetCtxLogLevel(const LogLevel level)
{
    level_ = level;
}

std::string HzMcuLogger::GetCtxId() const noexcept
{
    return ctxID_;
}

std::string HzMcuLogger::GetAppId() const noexcept
{
    return appID_;
}

}
}
}
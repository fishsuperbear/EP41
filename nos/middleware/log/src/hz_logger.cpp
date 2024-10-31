/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     ara_logger.cpp                                                         *
*  @brief    Implement of class Logger                                    *
*  Details.                                                                         *
*                                                                                   *
*  @version  0.0.0.1                                                                *
*                                                                                   *
*-----------------------------------------------------------------------------------*
*  Change History :                                                                 *
*  <Date>     | <Version> | <Author>       | <Description>                          *
*-----------------------------------------------------------------------------------*
*  2022/06/15 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#include "hz_logger.hpp"
#include "logstream.h"

#include "log_manager.hpp"
#include "log_ctx_impl.hpp"

namespace hozon {
namespace netaos {
namespace log {

// External function declaration
void Quit();

HzLogger::HzLogger(std::string ctxId, std::string ctxDescription, LogLevel ctxDefLogLevel)
:Logger()
{
    // Creat a log implement
    pImpl = std::make_unique<CtxImpl>(*this, ctxId, ctxDescription, ctxDefLogLevel);
}

HzLogger::~HzLogger()
{
    Quit();
}

LogStream HzLogger::LogCritical() noexcept
{
    return LogStream{LogLevel::kCritical, this};
}

LogStream HzLogger::LogError() noexcept
{
    return LogStream{LogLevel::kError, this};
}

LogStream HzLogger::LogWarn() noexcept
{
    return LogStream{LogLevel::kWarn, this};
}

LogStream HzLogger::LogInfo() noexcept
{
    return LogStream{LogLevel::kInfo, this};
}

LogStream HzLogger::LogDebug() noexcept
{
    return LogStream{LogLevel::kDebug, this};
}

LogStream HzLogger::LogTrace() noexcept
{
    return LogStream{LogLevel::kTrace, this};
}

bool HzLogger::IsOperationLog() noexcept
{
    return false;
}

bool HzLogger::IsEnabled(LogLevel level) noexcept
{
    return pImpl->IsEnabled(level);
}

bool HzLogger::SetLogLevel(const LogLevel level) noexcept
{
    ForceSetCtxLogLevel(level);
    return true;
}

void HzLogger::UpdateAppLogLevel(const LogLevel level) const noexcept
{
    pImpl->UpdateAppLogLevel(level);
}

void HzLogger::NormalSetCtxLogLevel(const LogLevel level) const noexcept
{
    pImpl->normalSetCtxLogLevel(level);
}

void HzLogger::ForceSetCtxLogLevel(const LogLevel level) const noexcept
{
    pImpl->forceSetCtxLogLevel(level);
}


LogLevel HzLogger::GetCtxLogLevel() const noexcept
{
    return pImpl->getCtxLogLevel();
}

LogLevel HzLogger::GetOutputLogLevel() const noexcept
{
    return pImpl->getOutputLogLevel();
}

std::string HzLogger::GetCtxId() const noexcept
{
    return std::move(pImpl->getCtxLogId());
}

const std::string HzLogger::GetCtxDescription() const noexcept
{
    return std::move(pImpl->getCtxLogDescription());
}

void HzLogger::LogOut(LogLevel level, const std::string& message)
{
    pImpl->LogOut(level, message);
}

}
}
}
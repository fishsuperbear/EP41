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
*  2023/04/06 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/
#include <iostream>
#include "hz_operation_logger.hpp"
#include "logstream.h"
#include "log_ctx_impl.hpp"
#include "log_manager.hpp"

#include "zmq_ipc/proto/log_server.pb.h"

namespace hozon {
namespace netaos {
namespace log {

namespace
{
    const std::string operation_log_service_name = "tcp://localhost:15777";
}

void Quit();

HzOperationLogger::HzOperationLogger(std::string ctxId, std::string ctxDescription, LogLevel ctxDefLogLevel)
:Logger(),
ctxID_(ctxId)
{
    pImpl = std::make_unique<CtxImpl>(*this, ctxId, ctxDescription, ctxDefLogLevel);
    client_ = std::make_unique<hozon::netaos::zmqipc::ZmqIpcClient>();
    client_->Init(operation_log_service_name);
}

HzOperationLogger::~HzOperationLogger()
{
    // std::cout << "~HzOperationLogger !!!" << std::endl;
    client_->Deinit();
    Quit();
}

LogStream HzOperationLogger::LogCritical() noexcept
{
    return LogStream{LogLevel::kCritical, this};
}

LogStream HzOperationLogger::LogError() noexcept
{
    return LogStream{LogLevel::kError, this};
}

LogStream HzOperationLogger::LogWarn() noexcept
{
    return LogStream{LogLevel::kWarn, this};
}

LogStream HzOperationLogger::LogInfo() noexcept
{
    return LogStream{LogLevel::kInfo, this};
}

LogStream HzOperationLogger::LogDebug() noexcept
{
    return LogStream{LogLevel::kDebug, this};
}

LogStream HzOperationLogger::LogTrace() noexcept
{
    return LogStream{LogLevel::kTrace, this};
}

bool HzOperationLogger::IsOperationLog() noexcept
{
    return true;
}

bool HzOperationLogger::IsEnabled(LogLevel level) noexcept
{
    return pImpl->IsEnabled(level);
}

bool HzOperationLogger::SetLogLevel(const LogLevel level) noexcept
{
    ForceSetCtxLogLevel(level);
    return true;
}

void HzOperationLogger::LogOut(LogLevel level, const std::string& message)
{
    // 没有进行initLogger时，无法输出operation日志，在这里做一个过滤
    std::string appID = HzLogManager::GetInstance()->getAppId();
    if (appID == "INVALID")
    {
        std::cout << "INVALID!!!" << std::endl;
        return;
    }

    // 输出到普通日志
    std::string msg = "[operation] [" + GetCtxId() + "] " + message;
    pImpl->LogOut(level, msg);

    try {
        // [appID] [ctxID] [level] [message]
        LogoutInfo info{};
        info.set_app_id(appID);
        info.set_ctx_id(ctxID_);
        info.set_log_level(static_cast<uint32_t>(level));
        info.set_message(message);

        std::string serializedData = info.SerializeAsString();
        client_->RequestAndForget(serializedData);
    }
    catch (const zmq::error_t& ex) {
        std::cout << " zmq Request error" << std::endl;
        std::cout << " this operation msg loss : " << message << std::endl;
    }
}

LogLevel HzOperationLogger::GetOutputLogLevel() const noexcept
{
    return pImpl->getOutputLogLevel();
}

void HzOperationLogger::UpdateAppLogLevel(const LogLevel level) const noexcept
{
    pImpl->UpdateAppLogLevel(level);
}

void HzOperationLogger::NormalSetCtxLogLevel(const LogLevel level) const noexcept
{
    pImpl->normalSetCtxLogLevel(level);
}

void HzOperationLogger::ForceSetCtxLogLevel(const LogLevel level) const noexcept
{
    pImpl->forceSetCtxLogLevel(level);
}

std::string HzOperationLogger::GetCtxId() const noexcept
{
    return std::move(pImpl->getCtxLogId());
}

}
}
}
/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     logger.h                                                     *
*  @brief    Define of class logger                                          *
*  Details.                                                                         *
*                                                                                   *
*  @version  0.0.0.1                                                                *
*                                                                                   *
*-----------------------------------------------------------------------------------*
*  Change History :                                                                 *
*  <Date>     | <Version> | <Author>       | <Description>                          *
*-----------------------------------------------------------------------------------*
*  2023/04/6 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#ifndef __HZ_OPERATION_LOGGER_HPP__
#define __HZ_OPERATION_LOGGER_HPP__

#include <atomic>
#include <memory>
#include <string>
#include <mutex>

#include "logger.h"
#include "log_ctx_impl.hpp"
#include "zmq_ipc/manager/zmq_ipc_client.h"

namespace hozon {
namespace netaos {
namespace log {


class HzOperationLogger final: public Logger {
 public:
    HzOperationLogger(std::string ctxId, std::string ctxDescription, LogLevel ctxDefLogLevel);

    ~HzOperationLogger();

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

    void UpdateAppLogLevel(const LogLevel level) const noexcept;

    void NormalSetCtxLogLevel(const LogLevel level) const noexcept;

    void ForceSetCtxLogLevel(const LogLevel level) const noexcept;

    std::string GetCtxId() const noexcept;

 private:
   std::unique_ptr<CtxImpl> pImpl;
   std::mutex mtx_;
   std::unique_ptr<hozon::netaos::zmqipc::ZmqIpcClient> client_;
   std::string ctxID_;
};


}
}
}

#endif  //__HZ_LOGGER_HPP__

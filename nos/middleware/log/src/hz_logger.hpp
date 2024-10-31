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
*  2022/06/15 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/

#ifndef __HZ_LOGGER_HPP__
#define __HZ_LOGGER_HPP__

#include <atomic>
#include <memory>
#include <string>

#include "logger.h"
#include "log_ctx_impl.hpp"

namespace hozon {
namespace netaos {
namespace log {

class HzLogger final: public Logger {
 public:
    HzLogger(std::string ctxId, std::string ctxDescription, LogLevel ctxDefLogLevel);

    ~HzLogger();

    virtual LogStream LogCritical() noexcept;

    virtual LogStream LogError() noexcept;

    virtual LogStream LogWarn() noexcept;

    virtual LogStream LogInfo() noexcept;
 
    virtual LogStream LogDebug() noexcept;

    virtual LogStream LogTrace() noexcept;

    virtual bool IsOperationLog() noexcept;
   
    virtual bool IsEnabled(LogLevel level) noexcept;

    virtual bool SetLogLevel(const LogLevel level) noexcept;

    void UpdateAppLogLevel(const LogLevel level) const noexcept;

    void NormalSetCtxLogLevel(const LogLevel level) const noexcept;

    void ForceSetCtxLogLevel(const LogLevel level) const noexcept;

    LogLevel GetCtxLogLevel() const noexcept;

    LogLevel GetOutputLogLevel() const noexcept;

   std::string GetCtxId() const noexcept;
    const std::string GetCtxDescription() const noexcept;
    void LogOut(LogLevel level, const std::string& message);

 private:
    std::unique_ptr<CtxImpl> pImpl;
};


}
}
}

#endif  //__HZ_LOGGER_HPP__

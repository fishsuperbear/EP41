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

#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <atomic>
#include <memory>
#include <cstdint>
namespace hozon {
namespace netaos {
namespace log {

static const std::uint32_t HZ_LOG2CONSOLE = (0x00000001 << 0);
static const std::uint32_t HZ_LOG2FILE = (0x00000001 << 1);
static const std::uint32_t HZ_LOG2LOGSERVICE = (0x00000001 << 2);

enum class LogLevel : uint8_t {
    kTrace = 0x00U,
    kDebug = 0x01U,
    kInfo = 0x02U,
    kWarn = 0x03U,
    kError = 0x04U,
    kCritical = 0x05U,
    kOff = 0x06U
};

class LogStream;
class Logger {
public:
    Logger(){}

    virtual ~Logger(){}

    virtual LogStream LogCritical()  noexcept = 0;
    virtual LogStream LogError() noexcept = 0;
    virtual LogStream LogWarn() noexcept = 0;
    virtual LogStream LogInfo() noexcept = 0;
    virtual LogStream LogDebug() noexcept = 0;
    virtual LogStream LogTrace() noexcept = 0;
    virtual bool IsOperationLog() noexcept = 0;
    virtual bool IsEnabled(LogLevel level) noexcept = 0;
    virtual bool SetLogLevel(const LogLevel level) noexcept = 0;
    virtual LogLevel GetOutputLogLevel() const noexcept = 0;
    virtual void LogOut(LogLevel level, const std::string& message) = 0;
    virtual std::string GetCtxId() const noexcept = 0;
};


}
}
}

#endif  //__LOGGER_H__

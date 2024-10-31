/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     logstream.h                                                            *
*  @brief    Define of class HzLogTrace                                             *
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

#ifndef __HZ_LOGSTREAM_H__
#define __HZ_LOGSTREAM_H__

#include <sstream>
#include <iomanip>
#include "logger.h"
#include "loglocation.h"
#include "logprecision.h"
#include "logfixed.h"

namespace hozon {
namespace netaos {
namespace log {

struct LogRawBuffer {
    const void * const buffer;
    uint16_t size;
};

struct LogHex8 {
    uint8_t value;
};

struct LogHex16 {
    uint16_t value;
};

struct LogHex32 {
    uint32_t value;
};

struct LogHex64 {
    uint64_t value;
};

struct Setprecision {
    explicit Setprecision(const int32_t& precision) : value(precision) {}
    int32_t value;
};
class LogStream final {
public:
    LogStream() = delete;

    LogStream(LogLevel level, Logger * selfLogger) noexcept;

    ~LogStream();

    LogStream(const LogStream &) = delete;
    LogStream &operator =(const LogStream &) = delete;
    LogStream(LogStream && other);
    LogStream &operator =(LogStream &&) = delete;

    LogStream &operator <<(bool value) noexcept;

    LogStream &operator <<(uint8_t value) noexcept;

    LogStream &operator <<(uint16_t value) noexcept;

    LogStream &operator <<(uint32_t value) noexcept;

    LogStream &operator <<(uint64_t value) noexcept;

    LogStream &operator <<(int8_t value) noexcept;

    LogStream &operator <<(int16_t value) noexcept;

    LogStream &operator <<(int32_t value) noexcept;

    LogStream &operator <<(int64_t value) noexcept;

    LogStream &operator <<(long long value) noexcept;

    LogStream &operator <<(unsigned long long value) noexcept;

    LogStream &operator <<(float value) noexcept;

    LogStream &operator <<(double value) noexcept;

    LogStream &operator <<(const LogRawBuffer &value) noexcept;

    LogStream &operator <<(const LogHex8 &value) noexcept;

    LogStream &operator <<(const LogHex16 &value) noexcept;

    LogStream &operator <<(const LogHex32 &value) noexcept;

    LogStream &operator <<(const LogHex64 &value) noexcept;

    LogStream &operator <<(const char * const value) noexcept;

    LogStream &operator <<(const std::string &value) noexcept;

    // Add source file location into the message
    LogStream &withLocation(const char* file, int line) noexcept;
    LogStream &withLocation(const SourceLocation& sourceLocation) noexcept;

    LogStream &withPrecision(int num) noexcept;
    LogStream &withPrecision(const DataPrecison& dataPrecison) noexcept;

    LogStream &withFixed() noexcept;
    LogStream &withFixed(const Fixed& fixed) noexcept;

    inline LogStream &operator <<(const SourceLocation& value) { return withLocation(value); }
    inline LogStream &operator <<(const DataPrecison& value) { return withPrecision(value); }
    inline LogStream &operator <<(const Fixed& fixed) { return withFixed(fixed); }

private:
    void LogOut();

    Logger * m_logger;
    LogLevel m_logLevel;
    std::ostringstream m_osLog;
    bool m_isOperationLog;
    bool m_isOutPut;
};

LogStream &operator <<(LogStream &out, LogLevel value) noexcept;

//LogStream &operator<<(LogStream &out, const ara::core::ErrorCode &value) noexcept;

constexpr LogHex32 loghex(uint32_t val) noexcept
{
    return LogHex32 { val };
}
constexpr LogHex64 loghex(uint64_t val) noexcept
{
    return LogHex64 { val };
}
inline LogStream &operator <<(LogStream &out, const void* const value) noexcept
{
    return (out << loghex(reinterpret_cast<std::uintptr_t>(value)));
}

}
}
}

#endif //__HZ_LOGSTREAM_H__

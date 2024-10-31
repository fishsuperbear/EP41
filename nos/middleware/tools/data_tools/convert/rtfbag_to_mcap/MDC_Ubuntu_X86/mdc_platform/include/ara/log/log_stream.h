/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Log stream head file
 * Create: 2021-7-12
 */
#ifndef ARA_LOG_LOG_STREAM_H
#define ARA_LOG_LOG_STREAM_H

#include <sstream>
#include <iomanip>

#include "ara/log/common.h"
#include "ara/log/logger.h"
#include "ara/core/string_view.h"
#include "ara/core/string.h"
#include "ara/core/error_code.h"

namespace ara {
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

struct LogBin8 {
    uint8_t value;
};

struct LogBin16 {
    uint16_t value;
};

struct LogBin32 {
    uint32_t value;
};

struct LogBin64 {
    uint64_t value;
};

struct Setprecision {
    explicit Setprecision(const int32_t& precision) noexcept : value(precision) {}
    int32_t value;
};
class LogStream final {
public:
    LogStream() = delete;

    LogStream(LogLevel logLevel, Logger &logger) noexcept;

    ~LogStream();

    LogStream(const LogStream &) = delete;
    LogStream &operator =(const LogStream &) = delete;
    LogStream(LogStream && other);
    LogStream &operator =(LogStream &&) = delete;

    void Flush() noexcept;
    void Fixed() noexcept;

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

    LogStream &operator <<(const LogBin8 &value) noexcept;

    LogStream &operator <<(const LogBin16 &value) noexcept;

    LogStream &operator <<(const LogBin32 &value) noexcept;

    LogStream &operator <<(const LogBin64 &value) noexcept;

    LogStream &operator <<(const ara::core::StringView value) noexcept;

    LogStream &operator<<(const char * const value) noexcept;

    LogStream &operator<<(const std::string &value) noexcept;

    LogStream &operator<<(const ara::core::String &value) noexcept;

    LogStream &operator<<(const Setprecision &value) noexcept;

    LogStream &operator<<(LogStream& (*const fun)(LogStream&)) noexcept;
private:
    void LogOut() noexcept;

    Logger &logger_;
    LogLevel logLevel_;
    LogReturnValue logRet_;
    std::ostringstream osLog_;
};

LogStream &operator <<(LogStream &out, LogLevel value) noexcept;

LogStream &operator<<(LogStream &out, const ara::core::ErrorCode &value) noexcept;

constexpr LogHex32 LogHex(uint32_t val) noexcept
{
    return LogHex32 { val };
}
constexpr LogHex64 LogHex(uint64_t val) noexcept
{
    return LogHex64 { val };
}
inline LogStream &operator <<(LogStream &out, const void* const value) noexcept
{
    return (out << LogHex(reinterpret_cast<std::uintptr_t>(value)));
}

inline LogStream& Fixed(LogStream &out) noexcept
{
    out.Fixed();
    return out;
}
}
}

#endif  // ARA_LOG_LOG_STREAM_H

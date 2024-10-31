/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     ara_LogStream.cpp                                                      *
*  @brief    Implement of class ara::log::LogStream                                 *
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

#include "logstream.h"
#include <iostream>

#include "log_ctx_impl.hpp"
#include <chrono>
#include "log_manager.hpp"
#include "hz_operation_logger.hpp"

namespace hozon {
namespace netaos {
namespace log {

LogStream::LogStream(LogLevel level, Logger * selfLogger) noexcept
    : m_logger{selfLogger},
      m_logLevel(level),
      m_isOperationLog(m_logger->IsOperationLog()),
      m_isOutPut(false)
{
    std::stringstream ss;

    if (m_isOperationLog) {
        m_osLog << ss.str() << "[";
        if (m_logger->GetOutputLogLevel() <= m_logLevel) {
            m_isOutPut = true;
        }
    }
    else {
        if (m_logger->GetOutputLogLevel() <= m_logLevel) {
            if (HzLogManager::GetInstance()->getPureLogFormat()) {
                m_osLog << ss.str();
            }
            else {
                // For display header info for each line of the log.
                m_osLog << ss.str() << "[" << m_logger->GetCtxId() << "] [";
            }
            m_isOutPut = true;
        }
    }
}

LogStream::~LogStream()
{
    if (m_isOutPut) {
        LogOut();
    }
}


LogStream::LogStream(LogStream && other)
:m_logger(other.m_logger)
,m_logLevel(other.m_logLevel)
,m_osLog(std::move(other.m_osLog))
{
}

LogStream &LogStream::operator << (bool value) noexcept
{
    if (m_isOutPut) {
        if (value) {
            m_osLog << "TRUE";
        }
        else {
            m_osLog << "FALSE";
        }
    }
    return *this;
}

LogStream &LogStream::operator << (uint8_t value) noexcept
{
    if (m_isOutPut) {
        m_osLog << static_cast<uint32_t>(value);
    }
    return *this;
}

LogStream &LogStream::operator <<(uint16_t value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator << (uint32_t value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(uint64_t value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(int8_t value) noexcept
{
    if (m_isOutPut) {
        m_osLog << static_cast<int32_t>(value);
    }
    return *this;
}

LogStream &LogStream::operator <<(int16_t value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(int32_t value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(int64_t value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(long long value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(unsigned long long value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(float value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(double value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator <<(const LogRawBuffer &value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value.buffer;
    }
    return *this;
}

LogStream &LogStream::operator <<(const LogHex8 &value) noexcept
{
    if (m_isOutPut) {
        m_osLog << std::setbase(16) << value.value;
    }
    return *this;
}

LogStream &LogStream::operator <<(const LogHex16 &value) noexcept
{
    if (m_isOutPut) {
        m_osLog << std::setbase(16) << value.value;
    }
    return *this;
}

LogStream &LogStream::operator <<(const LogHex32 &value) noexcept
{
    if (m_isOutPut) {
        m_osLog << std::setbase(16) << value.value;
    }
    return *this;
}

LogStream &LogStream::operator <<(const LogHex64 &value) noexcept
{
    if (m_isOutPut) {
        m_osLog << std::setbase(16) << value.value;
    }
    return *this;
}


LogStream &LogStream::operator<<(const char * const value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream &LogStream::operator<<(const std::string &value) noexcept
{
    if (m_isOutPut) {
        m_osLog << value;
    }
    return *this;
}

LogStream& LogStream::withLocation(const char* file, int line) noexcept
{
    m_osLog << file << "." << line << " ";
    return *this;
}

LogStream& LogStream::withLocation(const SourceLocation& sourceLocation) noexcept
{
    return withLocation(sourceLocation.fileName, sourceLocation.lineNo);
}

LogStream& LogStream::withPrecision(int num) noexcept
{
    m_osLog << std::setprecision(num);
    return *this;
}

LogStream& LogStream::withPrecision(const DataPrecison& dataPrecison) noexcept
{
    return withPrecision(dataPrecison.precision);
}

LogStream& LogStream::withFixed() noexcept
{
    m_osLog << std::fixed;
    return *this;
}

LogStream& LogStream::withFixed(const Fixed& fixed) noexcept
{
    return withFixed();
}

void LogStream::LogOut()
{
    m_logger->LogOut(m_logLevel, m_osLog.str());
}

}
}
}

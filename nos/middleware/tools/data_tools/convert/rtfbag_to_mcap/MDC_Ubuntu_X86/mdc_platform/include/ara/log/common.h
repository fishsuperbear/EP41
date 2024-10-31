/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 * Create: 2020/5/23.
 */

#ifndef LOGCOMMON_H
#define LOGCOMMON_H

#include <memory>
#include <iostream>

#include "securec.h"

namespace ara {
namespace log {
enum class LogMode : uint8_t {
    kRemote = 0x01U,
    kFile = 0x02U,
    kConsole = 0x04U
};

enum class LogLevel : uint8_t {
    kOff = 0x00U,
    kFatal = 0x01U,
    kError = 0x02U,
    kWarn = 0x03U,
    kInfo = 0x04U,
    kDebug = 0x05U,
    kVerbose = 0x06U
};

enum class LogReturnValue : int8_t {
    LOG_RETURN_LOGGING_DISABLE = -2,
    LOG_RETURN_ERROR = -1,
    LOG_RETURN_OK = 0,
    LOG_RETURN_TRUE = 1
};

/**
 * Client state representing the connection state of an external client.
 * @uptrace{SWS_LOG_00098}
 */
enum class ClientState : int8_t
{
    kUnknown = -1,  //< DLT back-end not up and running yet, state cannot be determined
    kNotConnected,  //< No remote client detected
    kConnected  //< Remote client is connected
};

inline std::ostream &operator <<(std::ostream &os, const LogLevel &level)
{
    switch (level) {
        case LogLevel::kOff:
            os << "off";
            break;
        case LogLevel::kFatal:
            os << "fatal";
            break;
        case LogLevel::kError:
            os << "error";
            break;
        case LogLevel::kWarn:
            os << "warn";
            break;
        case LogLevel::kInfo:
            os << "info";
            break;
        case LogLevel::kDebug:
            os << "debug";
            break;
        case LogLevel::kVerbose:
            os << "verbose";
            break;
        default:
            break;
    }
    return os;
}

inline LogMode operator|(const LogMode lhs, const LogMode rhs)
{
    return (static_cast<LogMode>(static_cast<typename std::underlying_type<LogMode>::type>(lhs) |
        static_cast<typename std::underlying_type<LogMode>::type>(rhs)));
}

inline LogMode operator&(const LogMode lhs, const LogMode rhs)
{
    return (static_cast<LogMode>(static_cast<typename std::underlying_type<LogMode>::type>(lhs) &
        static_cast<typename std::underlying_type<LogMode>::type>(rhs)));
}
}
}

#endif

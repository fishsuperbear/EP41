/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description:
 *      This file is the implement of class Log.
 *      Log will provide rtf log function.
 * Create: 2020-07-03
 * Notes: NA
 */
#ifndef RTF_LOG_H
#define RTF_LOG_H

#include <iostream>
#include <memory>
#include <string>

#include "ara/hwcommon/log/log.h"

namespace rtf {
class RtfLog {
public:
    using LogStream = ara::godel::common::log::LogStreamBuffer;
    using Log = ara::godel::common::log::Log;
    static void InitLog();
    static std::shared_ptr<Log>& GetInstance();
    static LogStream Verbose() noexcept;
    static LogStream Debug() noexcept;
    static LogStream Info() noexcept;
    static LogStream Warn() noexcept;
    static LogStream Error() noexcept;
    static LogStream Error(const std::string& logTag, const std::uint32_t& logLimit) noexcept;
private:
    static std::shared_ptr<Log> logger_;
    static std::once_flag logIsInitialized_;
};
}
#endif

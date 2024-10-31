/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: This class provides logging capability of the Ros-like interface.
 * Create: 2020-06-30
 */

#ifndef RTF_COM_UTILS_LOGGER_H
#define RTF_COM_UTILS_LOGGER_H

#include <memory>
#include "ara/hwcommon/log/log.h"

namespace rtf   {
namespace com   {
namespace utils {
class Logger {
public:
    using LogInterface = ara::godel::common::log::Log;
    using LogStream    = ara::godel::common::log::LogStreamBuffer;
    Logger();
    ~Logger() = default;
    static std::shared_ptr<Logger> GetInstance();
    LogStream Verbose() noexcept;
    LogStream Debug() noexcept;
    LogStream Info() noexcept;
    LogStream Warn() noexcept;
    LogStream Error() noexcept;
    LogStream Error(std::string const &logUUID, std::uint32_t logNumber, bool stateChange) noexcept;
    LogStream Fatal() noexcept;
private:
    /* AXIVION enable style AutosarC++19_03-A7.1.3 */
    std::shared_ptr<LogInterface> logger_;
};
} // namespace utils
} // namespace com
} // namespace rtf
#endif // RTF_COM_UTILS_LOGGER_H_
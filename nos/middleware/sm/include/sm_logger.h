/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: sm_logger.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 *
 */
#ifndef SM_LOGGER_H
#define SM_LOGGER_H

#include <iostream>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace sm {

class SMLogger
{
public:
    static SMLogger& GetInstance() {
        static SMLogger instance;
        return instance;
    }
    ~SMLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger>  GetLogger() const { return logger_; }

private:
    SMLogger(){
    };
    std::shared_ptr<hozon::netaos::log::Logger> logger_ {
        hozon::netaos::log::CreateLogger("SM", "NETAOS SM",
                        hozon::netaos::log::LogLevel::kInfo) };
};

#define SM_LOG_CRITICAL       (SMLogger::GetInstance().GetLogger()->LogCritical())
#define SM_LOG_ERROR          (SMLogger::GetInstance().GetLogger()->LogError())
#define SM_LOG_WARN           (SMLogger::GetInstance().GetLogger()->LogWarn())
#define SM_LOG_INFO           (SMLogger::GetInstance().GetLogger()->LogInfo())
#define SM_LOG_DEBUG          (SMLogger::GetInstance().GetLogger()->LogDebug())
#define SM_LOG_TRACE          (SMLogger::GetInstance().GetLogger()->LogTrace())

#define SERVER_HEAD " (SERVER) " <<__FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "
#define CLIENT_HEAD " (CLIENT) " <<__FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 <<  "(" << __LINE__ << ") | "
#define SM_SERVER_LOG_CRITICAL       (SMLogger::GetInstance().GetLogger()->LogCritical() << SERVER_HEAD)
#define SM_SERVER_LOG_ERROR          (SMLogger::GetInstance().GetLogger()->LogError() << SERVER_HEAD)
#define SM_SERVER_LOG_WARN           (SMLogger::GetInstance().GetLogger()->LogWarn() << SERVER_HEAD)
#define SM_SERVER_LOG_INFO           (SMLogger::GetInstance().GetLogger()->LogInfo() << SERVER_HEAD)
#define SM_SERVER_LOG_DEBUG          (SMLogger::GetInstance().GetLogger()->LogDebug() << SERVER_HEAD)
#define SM_SERVER_LOG_TRACE          (SMLogger::GetInstance().GetLogger()->LogTrace() << SERVER_HEAD)

#define SM_CLIENT_LOG_CRITICAL       (SMLogger::GetInstance().GetLogger()->LogCritical() << CLIENT_HEAD)
#define SM_CLIENT_LOG_ERROR          (SMLogger::GetInstance().GetLogger()->LogError() << CLIENT_HEAD)
#define SM_CLIENT_LOG_WARN           (SMLogger::GetInstance().GetLogger()->LogWarn() << CLIENT_HEAD)
#define SM_CLIENT_LOG_INFO           (SMLogger::GetInstance().GetLogger()->LogInfo() << CLIENT_HEAD)
#define SM_CLIENT_LOG_DEBUG          (SMLogger::GetInstance().GetLogger()->LogDebug() << CLIENT_HEAD)
#define SM_CLIENT_LOG_TRACE          (SMLogger::GetInstance().GetLogger()->LogTrace() << CLIENT_HEAD)
} // namespace sm
} // namespace netaos
} // namespace hozon
#endif
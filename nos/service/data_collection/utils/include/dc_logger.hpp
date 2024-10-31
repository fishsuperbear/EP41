/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_logger.h
 * @Date: 2023/08/07
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_INCLUDE_DC_LOGGER_H
#define MIDDLEWARE_TOOLS_DATA_COLLECT_INCLUDE_DC_LOGGER_H

#include <iostream>
#include <mutex>
#include <sys/syscall.h>
#include <unistd.h>

#include "log/include/logging.h"
#include "common/dc_macros.h"

namespace hozon {
namespace netaos {
namespace dc {

class DcLogger {
    typedef std::shared_ptr<DcLogger> Ptr;
   public:
    static DcLogger* GetInstance() {

//        if (nullptr == instance_)
//        {
//            std::lock_guard<std::mutex> lck(mtx_);
//            if (nullptr == instance_)
//            {
//                instance_ = std::shared_ptr<DcLogger>(new DcLogger());
//            }
//        }
//        return instance_;
        static DcLogger instance;
        return &instance;
    }

    ~DcLogger(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

   private:
    DcLogger() {
//        hozon::netaos::log::InitLogging("DC", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, (20 * 1024 * 1024));
        hozon::netaos::log::InitLogging("DC", "NETAOS DC", hozon::netaos::log::LogLevel::kInfo,  hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, (20 * 1024 * 1024));
        logger_ = hozon::netaos::log::CreateLogger("DC", "NETAOS DC", hozon::netaos::log::LogLevel::kInfo);
    };

    std::shared_ptr<hozon::netaos::log::Logger> logger_;

};


class DcOper {
    typedef std::shared_ptr<DcOper> Ptr;
   public:
    static DcOper* GetInstance() {
        static DcOper instance;
        return &instance;
    }

    ~DcOper(){};

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

   private:
    DcOper() {
//        hozon::netaos::log::InitLogging("DCOPER", "NETAOS DC", hozon::netaos::log::LogLevel::kDebug,  hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, (20 * 1024 * 1024));
        logger_ = hozon::netaos::log::CreateOperationLogger("DCOPER", "NETAOS DC", hozon::netaos::log::LogLevel::kDebug);
    };
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};

//DcLogger::Ptr DcLogger::instance_ = nullptr;
//std::mutex DcLogger::mtx_;

#define OPER_HEAD " (OPER) "<< "P" << (long int)syscall(__NR_getpid)<< " T" << (long int)syscall(__NR_gettid) << " " << strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "
#define DC_OPER_LOG_ERROR (DcOper::GetInstance()->GetLogger()->LogError() << OPER_HEAD)
#define DC_OPER_LOG_INFO (DcOper::GetInstance()->GetLogger()->LogInfo() << OPER_HEAD)

#define DC_LOG_CRITICAL (DcLogger::GetInstance()->GetLogger()->LogCritical())
#define DC_LOG_ERROR (DcLogger::GetInstance()->GetLogger()->LogError())
#define DC_LOG_WARN (DcLogger::GetInstance()->GetLogger()->LogWarn())
#define DC_LOG_INFO (DcLogger::GetInstance()->GetLogger()->LogInfo())
#define DC_LOG_DEBUG (DcLogger::GetInstance()->GetLogger()->LogDebug())
#define DC_LOG_TRACE (DcLogger::GetInstance()->GetLogger()->LogTrace())

#define SERVER_HEAD " (SERVER) "<< "P" << (long int)syscall(__NR_getpid)<< " T" << (long int)syscall(__NR_gettid) << " "  << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << "(" << __LINE__ << ") | "
#define CLIENT_HEAD " (CLIENT) " <<"P" << (long int)syscall(__NR_getpid)<< " T" << (long int)syscall(__NR_gettid) << " " << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << "(" << __LINE__ << ") | "
#define DC_SERVER_LOG_CRITICAL (DcLogger::GetInstance()->GetLogger()->LogCritical() << SERVER_HEAD)
#define DC_SERVER_LOG_ERROR (DcLogger::GetInstance()->GetLogger()->LogError() << SERVER_HEAD)
#define DC_SERVER_LOG_WARN (DcLogger::GetInstance()->GetLogger()->LogWarn() << SERVER_HEAD)
#define DC_SERVER_LOG_INFO (DcLogger::GetInstance()->GetLogger()->LogInfo() << SERVER_HEAD)
#define DC_SERVER_LOG_DEBUG (DcLogger::GetInstance()->GetLogger()->LogDebug() << SERVER_HEAD)
#define DC_SERVER_LOG_TRACE (DcLogger::GetInstance()->GetLogger()->LogTrace() << SERVER_HEAD)

#define DC_CLIENT_LOG_CRITICAL (DcLogger::GetInstance()->GetLogger()->LogCritical() << CLIENT_HEAD)
#define DC_CLIENT_LOG_ERROR (DcLogger::GetInstance()->GetLogger()->LogError() << CLIENT_HEAD)
#define DC_CLIENT_LOG_WARN (DcLogger::GetInstance()->GetLogger()->LogWarn() << CLIENT_HEAD)
#define DC_CLIENT_LOG_INFO (DcLogger::GetInstance()->GetLogger()->LogInfo() << CLIENT_HEAD)
#define DC_CLIENT_LOG_DEBUG (DcLogger::GetInstance()->GetLogger()->LogDebug() << CLIENT_HEAD)
#define DC_CLIENT_LOG_TRACE (DcLogger::GetInstance()->GetLogger()->LogTrace() << CLIENT_HEAD)


}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_INCLUDE_DC_LOGGER_H

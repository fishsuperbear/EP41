/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-09-20 11:22:54
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-22 16:44:51
 * @FilePath: /nos/service/ethstack/soc_to_mcu/include/INTRA_logger.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: cfg
 * Description: log
 * Created on: Feb 7, 2023
 *
 */

#ifndef SERVICE_SOC_MCU_INCLUDE_INTRA_LOGGER_H_
#define SERVICE_SOC_MCU_INCLUDE_INTRA_LOGGER_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace intra {
/* INTRA log class */
class IntraLogger {
   public:
    static IntraLogger& GetInstance() {
        static IntraLogger instance;
        return instance;
    }
    ~IntraLogger() {}
    enum class IntraLogLevelType { INTRA_VERBOSE = 0, INTRA_DEBUG = 1, INTRA_INFO = 2, INTRA_WARN = 3, INTRA_ERROR = 4, INTRA_FATAL = 5, INTRA_OFF = 6 };
    hozon::netaos::log::LogLevel CFGParseLogLevel(const int32_t logLevel) {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<IntraLogLevelType>(logLevel);
        switch (type) {
            case IntraLogLevelType::INTRA_VERBOSE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case IntraLogLevelType::INTRA_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case IntraLogLevelType::INTRA_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case IntraLogLevelType::INTRA_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case IntraLogLevelType::INTRA_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case IntraLogLevelType::INTRA_FATAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case IntraLogLevelType::INTRA_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "SOCMCU_SERVER",                                                              // the log id of application
                     std::string appDescription = "soc_mcu_server",                                                    // the log id of application
                     IntraLogLevelType appLogLevel = IntraLogLevelType::INTRA_INFO,                                    // the log level of application
                     std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
                     std::string directoryPath = "/opt/usr/log/soc_log/",                                                                 // the log file directory, active when output log to file
                     std::uint32_t maxLogFileNum = 10,                                                                 // the max number log file , active when output log to file
                     std::uint64_t maxSizeOfLogFile = 20                                                               // the max size of each  log file , active when output log to file
    ) {
        level_ = static_cast<int32_t>(appLogLevel);
        const hozon::netaos::log::LogLevel applevel = CFGParseLogLevel(static_cast<int32_t>(appLogLevel));
        hozon::netaos::log::InitLogging(appId, appDescription, applevel, outputMode, directoryPath, maxLogFileNum, maxSizeOfLogFile, true);
        logger_ = hozon::netaos::log::CreateLogger("INTRA", "NETAOS INTRA", CFGParseLogLevel(static_cast<int32_t>(appLogLevel)));
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

   private:
    IntraLogger() {}
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
    int level_ = IntraLogLevelType::INTRA_INFO;
};

#define INTRA_LOG_HEAD         \
    " pid:" << getpid() << " " \
            << "tid:" << (int64_t)syscall(__NR_gettid) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define INTRA_LOG_CRITICAL hozon::netaos::intra::IntraLogger::GetInstance().GetLogger()->LogCritical() << INTRA_LOG_HEAD
#define INTRA_LOG_ERROR hozon::netaos::intra::IntraLogger::GetInstance().GetLogger()->LogError() << INTRA_LOG_HEAD
#define INTRA_LOG_WARN hozon::netaos::intra::IntraLogger::GetInstance().GetLogger()->LogWarn() << INTRA_LOG_HEAD
#define INTRA_LOG_INFO hozon::netaos::intra::IntraLogger::GetInstance().GetLogger()->LogInfo() << INTRA_LOG_HEAD
#define INTRA_LOG_DEBUG hozon::netaos::intra::IntraLogger::GetInstance().GetLogger()->LogDebug() << INTRA_LOG_HEAD
#define INTRA_LOG_TRACE hozon::netaos::intra::IntraLogger::GetInstance().GetLogger()->LogTrace() << INTRA_LOG_HEAD

}  // namespace intra
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_SOC_MCU_INCLUDE_INTRA_LOGGER_H_

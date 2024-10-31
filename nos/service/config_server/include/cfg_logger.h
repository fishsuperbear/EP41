/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-10-07 17:47:39
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-10-16 16:08:55
 * @FilePath: /nos/service/config_server/include/cfg_logger.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: cfg
 * Description: log
 * Created on: Feb 7, 2023
 *
 */

#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_LOGGER_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_LOGGER_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace cfg {
/* CONFIG log class */
class ConfigLogger {
 public:
    static ConfigLogger& GetInstance() {
        static ConfigLogger instance;
        return instance;
    }
    ~ConfigLogger() {}
    enum class CFGLogLevelType { CFG_VERBOSE = 0, CFG_DEBUG = 1, CFG_INFO = 2, CFG_WARN = 3, CFG_ERROR = 4, CFG_FATAL = 5, CFG_OFF = 6 };
    hozon::netaos::log::LogLevel CFGParseLogLevel(const int32_t logLevel) {
        hozon::netaos::log::LogLevel level;
        const auto type = static_cast<CFGLogLevelType>(logLevel);
        switch (type) {
            case CFGLogLevelType::CFG_VERBOSE:
                level = hozon::netaos::log::LogLevel::kTrace;
                break;
            case CFGLogLevelType::CFG_DEBUG:
                level = hozon::netaos::log::LogLevel::kDebug;
                break;
            case CFGLogLevelType::CFG_INFO:
                level = hozon::netaos::log::LogLevel::kInfo;
                break;
            case CFGLogLevelType::CFG_WARN:
                level = hozon::netaos::log::LogLevel::kWarn;
                break;
            case CFGLogLevelType::CFG_ERROR:
                level = hozon::netaos::log::LogLevel::kError;
                break;
            case CFGLogLevelType::CFG_FATAL:
                level = hozon::netaos::log::LogLevel::kCritical;
                break;
            case CFGLogLevelType::CFG_OFF:
                level = hozon::netaos::log::LogLevel::kOff;
                break;
            default:
                level = hozon::netaos::log::LogLevel::kError;
                break;
        }
        return level;
    }

    // only process can use this function
    void InitLogging(std::string appId = "CFG_SERVER",                                                                 // the log id of application
                     std::string appDescription = "cfg_server",                                                        // the log id of application
                     CFGLogLevelType appLogLevel = CFGLogLevelType::CFG_INFO,                                          // the log level of application
                     std::uint32_t outputMode = hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
                     std::string directoryPath = "/opt/usr/log/soc_log/",                                                                 // the log file directory, active when output log to file
                     std::uint32_t maxLogFileNum = 10,                                                                 // the max number log file , active when output log to file
                     std::uint64_t maxSizeOfLogFile = 20                                                               // the max size of each  log file , active when output log to file
    ) {
        const hozon::netaos::log::LogLevel applevel = CFGParseLogLevel(static_cast<int32_t>(appLogLevel));
        hozon::netaos::log::InitLogging(appId, appDescription, applevel, outputMode, directoryPath, maxLogFileNum, maxSizeOfLogFile, true);
        logger_ = hozon::netaos::log::CreateLogger("CFG", "NETAOS CFG", applevel);
    }

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

 private:
    ConfigLogger() {}
    std::shared_ptr<hozon::netaos::log::Logger> logger_;
};

#define CONFIG_LOG_HEAD        \
    " pid:" << getpid() << " " \
            << "tid:" << (int64_t)syscall(__NR_gettid) << " " << strrchr(__FILE__, '/') + 1 << "@" << __FUNCTION__ << "(" << __LINE__ << ") | "

#define CONFIG_LOG_CRITICAL hozon::netaos::cfg::ConfigLogger::GetInstance().GetLogger()->LogCritical() << CONFIG_LOG_HEAD
#define CONFIG_LOG_ERROR hozon::netaos::cfg::ConfigLogger::GetInstance().GetLogger()->LogError() << CONFIG_LOG_HEAD
#define CONFIG_LOG_WARN hozon::netaos::cfg::ConfigLogger::GetInstance().GetLogger()->LogWarn() << CONFIG_LOG_HEAD
#define CONFIG_LOG_INFO hozon::netaos::cfg::ConfigLogger::GetInstance().GetLogger()->LogInfo() << CONFIG_LOG_HEAD
#define CONFIG_LOG_DEBUG hozon::netaos::cfg::ConfigLogger::GetInstance().GetLogger()->LogDebug() << CONFIG_LOG_HEAD
#define CONFIG_LOG_TRACE hozon::netaos::cfg::ConfigLogger::GetInstance().GetLogger()->LogTrace() << CONFIG_LOG_HEAD

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_LOGGER_H_

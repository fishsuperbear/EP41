/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-07-19 16:28:59
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-10-16 16:08:45
 * @FilePath: /nos/middleware/cfg/include/cfg_logger.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: cfg
 * Description: log定义
 * Created on: Feb 7, 2023
 *
 */

#ifndef MIDDLEWARE_CFG_INCLUDE_CFG_LOGGER_H_
#define MIDDLEWARE_CFG_INCLUDE_CFG_LOGGER_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <memory>

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

    std::shared_ptr<hozon::netaos::log::Logger> GetLogger() const { return logger_; }

 private:
    ConfigLogger() {}
    std::shared_ptr<hozon::netaos::log::Logger> logger_{hozon::netaos::log::CreateLogger("CFG", "NETAOS CFG", hozon::netaos::log::LogLevel::kInfo)};
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
#endif  // MIDDLEWARE_CFG_INCLUDE_CFG_LOGGER_H_

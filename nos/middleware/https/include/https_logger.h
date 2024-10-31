/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: Https server loger
 */

#pragma once

#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace https {

/* Fm log class */
class HttpsLogger {
 public:
  HttpsLogger() : logger_(nullptr){};
  virtual ~HttpsLogger(){};

  enum class HttpsLogLevelType {
    HTTPSLOG_KOFF = 0,
    HTTPSLOG_CRITICAL = 1,
    HTTPSLOG_ERROR = 2,
    HTTPSLOG_WARN = 3,
    HTTPSLOG_INFO = 4,
    HTTPSLOG_DEBUG = 5,
    HTTPSLOG_TRACE = 6
  };

  hozon::netaos::log::LogLevel HttpsParseLogLevel(const int32_t logLevel) {
    hozon::netaos::log::LogLevel level;
    const auto type = static_cast<HttpsLogLevelType>(logLevel);
    switch (type) {
      case HttpsLogLevelType::HTTPSLOG_KOFF:
        level = hozon::netaos::log::LogLevel::kOff;
        break;
      case HttpsLogLevelType::HTTPSLOG_CRITICAL:
        level = hozon::netaos::log::LogLevel::kCritical;
        break;
      case HttpsLogLevelType::HTTPSLOG_ERROR:
        level = hozon::netaos::log::LogLevel::kError;
        break;
      case HttpsLogLevelType::HTTPSLOG_WARN:
        level = hozon::netaos::log::LogLevel::kWarn;
        break;
      case HttpsLogLevelType::HTTPSLOG_INFO:
        level = hozon::netaos::log::LogLevel::kInfo;
        break;
      case HttpsLogLevelType::HTTPSLOG_DEBUG:
        level = hozon::netaos::log::LogLevel::kDebug;
        break;
      case HttpsLogLevelType::HTTPSLOG_TRACE:
        level = hozon::netaos::log::LogLevel::kTrace;
        break;
      default:
        level = hozon::netaos::log::LogLevel::kError;
        break;
    }
    return level;
  }

  // only process can use this function
  void InitLogging(
      std::string appId = "HTTPS",  // the log id of application
      std::string appDescription =
          "default application",  // the log id of application
      HttpsLogLevelType appLogLevel =
          HttpsLogLevelType::HTTPSLOG_INFO,  // the log level of
                                                     // application
      std::uint32_t outputMode =
          hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
      std::string directoryPath =
          "/opt/usr/log/soc_log/",  // the log file directory, active when output log to file
      std::uint32_t maxLogFileNum =
          10,  // the max number log file , active when output log to file
      std::uint64_t maxSizeOfLogFile =
          (20 * 1024 * 1024)  // the max size of each  log file , active when
                              // output log to file
  ) {
    level_ = static_cast<int32_t>(appLogLevel);
    const hozon::netaos::log::LogLevel applevel =
        HttpsParseLogLevel(static_cast<int32_t>(appLogLevel));
    hozon::netaos::log::InitLogging(appId, appDescription, applevel, outputMode,
                                    directoryPath, maxLogFileNum,
                                    maxSizeOfLogFile);
  }

  // context regist diagserver
  void CreateLogger(const std::string ctxId) {
    const hozon::netaos::log::LogLevel level = HttpsParseLogLevel(level_);
    std::string ctxIdView(ctxId.c_str());
    std::string ctxDescription(ctxId + " Loger");
    std::string ctxDescView(ctxDescription.c_str());
    auto logger =
        hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
    logger_ = logger;
  }

  static HttpsLogger& GetInstance() {
    static HttpsLogger instance;
    return instance;
  }

  std::shared_ptr<hozon::netaos::log::Logger> HttpsGetLogger() const {
    return logger_;
  }

  int32_t getLogLevel() { return level_; }

  void setLogLevel(int32_t level) { level_ = level; }

 private:
  HttpsLogger(const HttpsLogger&);
  HttpsLogger& operator=(const HttpsLogger&);

 public:
  std::shared_ptr<hozon::netaos::log::Logger> logger_;

 private:
  int32_t level_ =
      static_cast<int32_t>(HttpsLogLevelType::HTTPSLOG_INFO);
};

#define HTTPS_HEAD                             \
  "pid: " << (long int)syscall(__NR_getpid) \
          << " tid: " << (long int)syscall(__NR_gettid) << " | "
#define HTTPS_CRITICAL                                                           \
  (HttpsLogger::GetInstance().HttpsGetLogger()->LogCritical() \
   << HTTPS_HEAD)
#define HTTPS_ERROR                                                           \
  (HttpsLogger::GetInstance().HttpsGetLogger()->LogError() \
   << HTTPS_HEAD)
#define HTTPS_WARN                                                           \
  (HttpsLogger::GetInstance().HttpsGetLogger()->LogWarn() \
   << HTTPS_HEAD)
#define HTTPS_INFO                                                           \
  (HttpsLogger::GetInstance().HttpsGetLogger()->LogInfo() \
   << HTTPS_HEAD)
#define HTTPS_DEBUG                                                           \
  (HttpsLogger::GetInstance().HttpsGetLogger()->LogDebug() \
   << HTTPS_HEAD)
#define HTTPS_TRACE                                                           \
  (HttpsLogger::GetInstance().HttpsGetLogger()->LogTrace() \
   << HTTPS_HEAD)

}  // namespace Https
}  // namespace netaos
}  // namespace hozon
// end of file

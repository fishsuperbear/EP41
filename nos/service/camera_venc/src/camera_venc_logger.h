/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: camera venc loger
 */

#ifndef CAMERA_VENC_LOGGER_H
#define CAMERA_VENC_LOGGER_H
#pragma once

#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

class CameraVencLogger {
 public:
  CameraVencLogger() : logger_(nullptr){};
  virtual ~CameraVencLogger(){};

  enum class CameraVencLogLevelType {
    KOFF = 0,
    CRITICAL = 1,
    ERROR = 2,
    WARN = 3,
    INFO = 4,
    DEBUG = 5,
    TRACE = 6
  };

  hozon::netaos::log::LogLevel CameraVencParseLogLevel(const int32_t logLevel) {
    hozon::netaos::log::LogLevel level;
    const auto type = static_cast<CameraVencLogLevelType>(logLevel);
    switch (type) {
      case CameraVencLogLevelType::KOFF:
        level = hozon::netaos::log::LogLevel::kOff;
        break;
      case CameraVencLogLevelType::CRITICAL:
        level = hozon::netaos::log::LogLevel::kCritical;
        break;
      case CameraVencLogLevelType::ERROR:
        level = hozon::netaos::log::LogLevel::kError;
        break;
      case CameraVencLogLevelType::WARN:
        level = hozon::netaos::log::LogLevel::kWarn;
        break;
      case CameraVencLogLevelType::INFO:
        level = hozon::netaos::log::LogLevel::kInfo;
        break;
      case CameraVencLogLevelType::DEBUG:
        level = hozon::netaos::log::LogLevel::kDebug;
        break;
      case CameraVencLogLevelType::TRACE:
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
      std::string appId = "CAMV",  // the log id of application
      std::string appDescription =
          "camera venc application",  // the log id of application
      CameraVencLogLevelType appLogLevel =
          CameraVencLogLevelType::ERROR,  // the log level of
                                     // application
      std::uint32_t outputMode =
          hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
      std::string directoryPath =
          "/opt/usr/log/soc_log/",  // the log file directory, active when output log to file
      std::uint32_t maxLogFileNum =
          10,  // the max number log file , active when output log to file
      std::uint64_t maxSizeOfLogFile =
          20  // the max size of each  log file , active when
                              // output log to file
  ) {
    level_ = static_cast<int32_t>(appLogLevel);
    const hozon::netaos::log::LogLevel applevel =
        CameraVencParseLogLevel(static_cast<int32_t>(appLogLevel));
    hozon::netaos::log::InitLogging(appId, appDescription, applevel, outputMode,
                                    directoryPath, maxLogFileNum,
                                    maxSizeOfLogFile);
  }

  // context regist diagserver
  void CreateLogger(const std::string ctxId) {
    const hozon::netaos::log::LogLevel level = CameraVencParseLogLevel(level_);
    std::string ctxIdView(ctxId.c_str());
    std::string ctxDescription(ctxId + " Loger");
    std::string ctxDescView(ctxDescription.c_str());
    auto logger =
        hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
    logger_ = logger;
  }

  static CameraVencLogger& GetInstance() {
    static CameraVencLogger instance;
    return instance;
  }

  std::shared_ptr<hozon::netaos::log::Logger> CameraVencGetLogger() const {
    return logger_;
  }

  int32_t getLogLevel() { return level_; }

  void setLogLevel(int32_t level) { level_ = level; }

 private:
  CameraVencLogger(const CameraVencLogger&);
  CameraVencLogger& operator=(const CameraVencLogger&);

 public:
  std::shared_ptr<hozon::netaos::log::Logger> logger_;

 private:
  int32_t level_ = static_cast<int32_t>(CameraVencLogLevelType::ERROR);
};

#define CAMV_HEAD                             \
  "P" << (long int)syscall(__NR_getpid) \
      << " T" << (long int)syscall(__NR_gettid) << " " << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "
#define CAMV_CRITICAL \
  (CameraVencLogger::GetInstance().CameraVencGetLogger()->LogCritical() << CAMV_HEAD)
#define CAMV_ERROR \
  (CameraVencLogger::GetInstance().CameraVencGetLogger()->LogError() << CAMV_HEAD)
#define CAMV_WARN \
  (CameraVencLogger::GetInstance().CameraVencGetLogger()->LogWarn() << CAMV_HEAD)
#define CAMV_INFO \
  (CameraVencLogger::GetInstance().CameraVencGetLogger()->LogInfo() << CAMV_HEAD)
#define CAMV_DEBUG \
  (CameraVencLogger::GetInstance().CameraVencGetLogger()->LogDebug() << CAMV_HEAD)
#define CAMV_TRACE \
  (CameraVencLogger::GetInstance().CameraVencGetLogger()->LogTrace() << CAMV_HEAD)

}  // namespace codec
}  // namespace netaos
}  // namespace hozon

#endif
// end of file

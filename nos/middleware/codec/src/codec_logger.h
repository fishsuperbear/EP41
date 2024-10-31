/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: codec server loger
 */

#ifndef CODEC_LOGGER_H_
#define CODEC_LOGGER_H_
#pragma once

#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>

#include "log/include/logging.h"

namespace hozon {
namespace netaos {
namespace codec {

/* Fm log class */
class CodecLogger {
 public:
  CodecLogger() : logger_(nullptr){};
  virtual ~CodecLogger(){};

  enum class CodecLogLevelType {
    KOFF = 0,
    CRITICAL = 1,
    ERROR = 2,
    WARN = 3,
    INFO = 4,
    DEBUG = 5,
    TRACE = 6
  };

  hozon::netaos::log::LogLevel CodecParseLogLevel(const int32_t logLevel) {
    hozon::netaos::log::LogLevel level;
    const auto type = static_cast<CodecLogLevelType>(logLevel);
    switch (type) {
      case CodecLogLevelType::KOFF:
        level = hozon::netaos::log::LogLevel::kOff;
        break;
      case CodecLogLevelType::CRITICAL:
        level = hozon::netaos::log::LogLevel::kCritical;
        break;
      case CodecLogLevelType::ERROR:
        level = hozon::netaos::log::LogLevel::kError;
        break;
      case CodecLogLevelType::WARN:
        level = hozon::netaos::log::LogLevel::kWarn;
        break;
      case CodecLogLevelType::INFO:
        level = hozon::netaos::log::LogLevel::kInfo;
        break;
      case CodecLogLevelType::DEBUG:
        level = hozon::netaos::log::LogLevel::kDebug;
        break;
      case CodecLogLevelType::TRACE:
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
      std::string appId = "DEFAULT_APP",  // the log id of application
      std::string appDescription =
          "default application",  // the log id of application
      CodecLogLevelType appLogLevel =
          CodecLogLevelType::ERROR,  // the log level of
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
        CodecParseLogLevel(static_cast<int32_t>(appLogLevel));
    hozon::netaos::log::InitLogging(appId, appDescription, applevel, outputMode,
                                    directoryPath, maxLogFileNum,
                                    maxSizeOfLogFile);
  }

  // context regist diagserver
  void CreateLogger(const std::string ctxId) {
    const hozon::netaos::log::LogLevel level = CodecParseLogLevel(level_);
    std::string ctxIdView(ctxId.c_str());
    std::string ctxDescription(ctxId + " Loger");
    std::string ctxDescView(ctxDescription.c_str());
    auto logger =
        hozon::netaos::log::CreateLogger(ctxIdView, ctxDescView, level);
    logger_ = logger;
  }

  static CodecLogger& GetInstance() {
    static CodecLogger instance;
    return instance;
  }

  std::shared_ptr<hozon::netaos::log::Logger> CodecGetLogger() const {
    return logger_;
  }

  int32_t getLogLevel() { return level_; }

  void setLogLevel(int32_t level) { level_ = level; }

 private:
  CodecLogger(const CodecLogger&);
  CodecLogger& operator=(const CodecLogger&);

 public:
  std::shared_ptr<hozon::netaos::log::Logger> logger_;

 private:
  int32_t level_ = static_cast<int32_t>(CodecLogLevelType::ERROR);
};

#define CODEC_HEAD                             \
  "P" << (long int)syscall(__NR_getpid) \
      << " T" << (long int)syscall(__NR_gettid) << " " << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << ":" << __LINE__ << " | "
#define CODEC_CRITICAL \
  (CodecLogger::GetInstance().CodecGetLogger()->LogCritical() << CODEC_HEAD)
#define CODEC_ERROR \
  (CodecLogger::GetInstance().CodecGetLogger()->LogError() << CODEC_HEAD)
#define CODEC_WARN \
  (CodecLogger::GetInstance().CodecGetLogger()->LogWarn() << CODEC_HEAD)
#define CODEC_INFO \
  (CodecLogger::GetInstance().CodecGetLogger()->LogInfo() << CODEC_HEAD)
#define CODEC_DEBUG \
  (CodecLogger::GetInstance().CodecGetLogger()->LogDebug() << CODEC_HEAD)
#define CODEC_TRACE \
  (CodecLogger::GetInstance().CodecGetLogger()->LogTrace() << CODEC_HEAD)

}  // namespace codec
}  // namespace netaos
}  // namespace hozon

#endif
// end of file

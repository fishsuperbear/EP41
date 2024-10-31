#pragma once

#include <memory>

#include "framework/log_interface/common.h"
#include "framework/log_interface/group_logger_base.h"
#include "framework/log_interface/log_stream.h"
#include "spdlog/spdlog.h"

namespace netaos {
namespace framework {
namespace loginterface {

class LogMessage {
 public:
  LogMessage(const char* file, int line,
             std::shared_ptr<GroupLoggerBase> logger, LogLevel level)
      : _file(file), _line(line), _logger(std::move(logger)), _level(level) {}

  ~LogMessage() noexcept { log_it(); }

  LogStream& stream() { return _stream; }

 protected:
  void log_it() {
    _logger->log(_file, _line, _level, _stream.buffer().data(),
                 _stream.buffer().size());
  }

 protected:
  const char* _file;
  int _line;
  std::shared_ptr<GroupLoggerBase> _logger;
  LogLevel _level;
  LogStream _stream;
};

class LogMessageDebug : public LogMessage {
 public:
  LogMessageDebug(const char* file, int line,
                  std::shared_ptr<GroupLoggerBase> logger)
      : LogMessage(file, line, logger, LogLevel::DEBUG) {}
};

class LogMessageInfo : public LogMessage {
 public:
  LogMessageInfo(const char* file, int line,
                 std::shared_ptr<GroupLoggerBase> logger)
      : LogMessage(file, line, logger, LogLevel::INFO) {}
};

class LogMessageWarning : public LogMessage {
 public:
  LogMessageWarning(const char* file, int line,
                    std::shared_ptr<GroupLoggerBase> logger)
      : LogMessage(file, line, logger, LogLevel::WARNING) {}
};

class LogMessageError : public LogMessage {
 public:
  LogMessageError(const char* file, int line,
                  std::shared_ptr<GroupLoggerBase> logger)
      : LogMessage(file, line, logger, LogLevel::ERROR) {}
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line,
                  std::shared_ptr<GroupLoggerBase> logger)
      : LogMessage(file, line, logger, LogLevel::FATAL) {}

  virtual ~LogMessageFatal() noexcept {
    spdlog::log(spdlog::source_loc{_file, _line, ""},
                netaos_to_spdlog_level(_level),
                spdlog::string_view_t(_stream.buffer().data(),
                                      _stream.buffer().size()));
    log_it();
    _logger->flush();
    std::abort();
  }
};

}  // namespace loginterface
}  // namespace framework
}  // namespace netaos

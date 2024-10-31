#pragma once

#include "framework/log_interface/common.h"
#include "framework/log_interface/group_logger_base.h"
#include "spdlog/async_logger.h"
#include "spdlog/spdlog.h"

namespace netaos {
namespace framework {
namespace logfile {

using logger_ptr = std::shared_ptr<spdlog::logger>;
using GroupLoggerBase = netaos::framework::loginterface::GroupLoggerBase;

class GroupLoggerFile : public GroupLoggerBase {
 public:
  GroupLoggerFile(const std::string& name, logger_ptr logger,
                  logger_ptr err_logger,
                  spdlog::level::level_enum level = spdlog::level::info,
                  bool flush_on_error = true);

  static std::shared_ptr<GroupLoggerFile> create(
      const std::string& name, logger_ptr logger, logger_ptr err_logger,
      spdlog::level::level_enum level = spdlog::level::info,
      bool flush_on_error = true) {
    return std::make_shared<GroupLoggerFile>(name, logger, err_logger, level,
                                             flush_on_error);
  }

  void flush() override;
  void set_level(spdlog::level::level_enum level) override;
  void log(const char* filename_in, int line_in,
           netaos::framework::LogLevel level, const char* data,
           size_t size) override;

  bool should_log(spdlog::level::level_enum level) const;

 private:
  logger_ptr _logger;  // in stdout, it is sync, in file output, it is async, so
                       // use base class
  logger_ptr _err_logger;  // error level do not use async logger
};

}  // namespace logfile
}  // namespace framework
}  // namespace netaos

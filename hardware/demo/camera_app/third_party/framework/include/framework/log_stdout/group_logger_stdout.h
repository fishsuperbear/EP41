#pragma once

#include "framework/log_interface/common.h"
#include "framework/log_interface/group_logger_base.h"
#include "spdlog/async_logger.h"
#include "spdlog/spdlog.h"

namespace netaos {
namespace framework {
namespace logstdout {

using logger_ptr = std::shared_ptr<spdlog::logger>;
using GroupLoggerBase = netaos::framework::loginterface::GroupLoggerBase;

class GroupLoggerStdout : public GroupLoggerBase {
 public:
  GroupLoggerStdout(const std::string& name, logger_ptr logger,
                    spdlog::level::level_enum level = spdlog::level::info);

  static std::shared_ptr<GroupLoggerStdout> create(
      const std::string& name, logger_ptr logger,
      spdlog::level::level_enum level = spdlog::level::info) {
    return std::make_shared<GroupLoggerStdout>(name, logger, level);
  }

  void flush() override;
  void set_level(spdlog::level::level_enum level) override;
  void log(const char* filename_in, int line_in,
           netaos::framework::LogLevel level, const char* data,
           size_t size) override;

  bool should_log(spdlog::level::level_enum level) const;

 private:
  logger_ptr _logger;
};

}  // namespace logstdout
}  // namespace framework
}  // namespace netaos

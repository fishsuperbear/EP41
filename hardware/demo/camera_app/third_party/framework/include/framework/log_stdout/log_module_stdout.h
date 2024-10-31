#pragma once

#include <map>
#include <memory>

#include "framework/common/macros.h"
#include "framework/concurrency/concurrent_hashmap.h"
#include "framework/log_interface/common.h"
#include "framework/log_stdout/group_logger_stdout.h"
#include "gflags/gflags.h"

DECLARE_uint32(async_log_threadnum);

namespace netaos {
namespace framework {
namespace logstdout {

class LogModuleStdout {
 public:
  LogModuleStdout();
  ~LogModuleStdout(){};
  std::shared_ptr<GroupLoggerStdout> create_console_logger(
      const std::string& name, LogLevel level);

 private:
  void create_console_sinks();

 private:
  spdlog::sink_ptr _console_sink;
  const std::string _pattern = "%L%m%d %H:%M:%S.%f %t %s:%#] [%n] %v";
};

}  // namespace logstdout
}  // namespace framework
}  // namespace netaos

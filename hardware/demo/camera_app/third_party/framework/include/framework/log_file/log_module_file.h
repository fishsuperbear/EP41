#pragma once

#include <map>
#include <memory>

#include "framework/common/macros.h"
#include "framework/concurrency/concurrent_hashmap.h"
#include "framework/log_file/group_logger_file.h"
#include "framework/log_interface/common.h"
#include "gflags/gflags.h"

DECLARE_string(log_directory);
DECLARE_uint32(async_log_threadnum);
DECLARE_uint32(max_log_MB);
DECLARE_uint32(max_log_files);
DECLARE_uint32(sec_log_interval);

namespace netaos {
namespace framework {
namespace logfile {

#define DEFAULT_QUEUE_SIZE (10 * 1024)

class LogModuleFile {
 public:
  LogModuleFile(const std::string& log_dir = FLAGS_log_directory,
                const std::string& logfile_prefix = "log",
                size_t queue_size = DEFAULT_QUEUE_SIZE);
  ~LogModuleFile(){};

  std::shared_ptr<GroupLoggerFile> create_file_logger(const std::string& name,
                                                      LogLevel level);
  void set_log_dir(const std::string& log_dir);
  void set_queue_size(size_t queue_size);
  void set_logfile_prefix(const std::string& logfile_prefix);

 private:
  std::string log_filename() const;
  std::string log_symlinkname() const;
  void create_file_sinks();

 private:
  std::string _log_dir;
  std::string _logfile_prefix;

  spdlog::sink_ptr _sink;
  spdlog::sink_ptr _err_sink;

  size_t _max_file_size;
  size_t _max_files_num;
  size_t _queue_size{DEFAULT_QUEUE_SIZE};
  size_t _flush_interval_sec;

  const std::string _pattern = "%L%m%d %H:%M:%S.%f %t %s:%#] [%n] %v";
};

}  // namespace logfile
}  // namespace framework
}  // namespace netaos

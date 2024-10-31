#pragma once

#include <map>
#include <memory>

#include "framework/common/macros.h"
#include "framework/concurrency/concurrent_hashmap.h"
#include "framework/log_file/group_logger_file.h"
#include "framework/log_file/log_module_file.h"
#include "framework/log_interface/common.h"
#include "framework/log_interface/group_logger_base.h"
#include "framework/log_interface/level.h"
#include "framework/log_stdout/group_logger_stdout.h"
#include "framework/log_stdout/log_module_stdout.h"
#include "gflags/gflags.h"

DECLARE_int32(log_to);
DECLARE_int32(min_log_level);

namespace netaos {
namespace framework {

enum class LogTo : int { STDOUT = 1, FILE, SERVER };

#define LOG_TO_STDOUT (FLAGS_log_to == (int)LogTo::STDOUT)
#define LOG_TO_FILE (FLAGS_log_to == (int)LogTo::FILE)
#define LOG_TO_SERVER (FLAGS_log_to == (int)LogTo::SERVER)

class ModuleLogConfig {
 public:
  void set_switch(bool sw) { _switch = sw; }

  bool get_switch() const { return _switch; }

  void set_level(LogLevel level) { _level = level; }

  LogLevel get_level() const { return _level; }

  std::string log_level_des() const {
    return loginterface::get_level_desc(_level);
  }

  inline bool operator==(const ModuleLogConfig& rhs) const {
    return ((_switch == rhs._switch) && (_level == rhs._level));
  }

 private:
  bool _switch;
  LogLevel _level;
};

namespace loginterface {

struct LogConfig {
  netaos::framework::ModuleLogConfig config;
  std::shared_ptr<GroupLoggerBase> logger;

  inline bool operator==(const LogConfig& rhs) const {
    return (config == rhs.config) && (logger == rhs.logger);
  }
};

/*****************************************************************************/
/* Todo: use ConcurrentHashMap to increase performance                       */
/*****************************************************************************/
class ModuleMapImpl final {
 public:
  using value_type = std::map<std::string, LogConfig>::value_type;

  inline bool insert_or_assign(const std::string& k, const LogConfig& conf) {
    return _map.insert_or_assign(k, conf);
  }

  inline bool insert(const std::string& k, const LogConfig& conf) {
    return _map.insert(k, conf);
  }

  inline bool assign(const std::string& k, const LogConfig& conf) {
    return _map.assign(k, conf);
  }

  inline bool assign_if_equal(const std::string& k, const LogConfig& expected,
                              const LogConfig& desired) {
    return _map.assign_if_equal(k, expected, desired);
  }

  inline void for_each(std::function<void(value_type&)> action) {
    _map.for_each(action);
  }

  inline bool contains(const std::string& k) const { return _map.contains(k); }

  inline bool get(const std::string& k, LogConfig* conf) const {
    return _map.get(k, conf);
  }

 private:
  netaos::framework::concurrency::ConcurrentHashMap<std::string, LogConfig>
      _map;
};

using LogModuleFile = netaos::framework::logfile::LogModuleFile;
using LogModuleStdout = netaos::framework::logstdout::LogModuleStdout;

class LogModule {
 public:
  ~LogModule(){};

  std::shared_ptr<GroupLoggerBase> get_logger(const std::string& name);
  void set_default_level(const std::string& level_des);
  bool valid_module(const std::string& name) const;
  bool set_switch(bool sw);
  bool set_switch(const std::string& name, bool sw);
  bool set_switch(const std::string& name, const std::string& sw);
  bool set_switch(const std::string& name, const std::string& sw,
                  std::ostream& out);
  bool get_switch(const std::string& name) const;
  bool set_level(const std::string& level_des);
  bool set_level(const std::string& name, const std::string& level_des);
  bool set_level(const std::string& name, LogLevel log_level,
                 const std::string& log_level_des, std::ostream& out);
  LogLevel get_level_by_desc(const std::string& level_des) const;
  LogLevel get_level_by_module(const std::string& module_name) const;
  bool get_module() const;
  bool get_config() const;
  bool get_config(std::ostream& out) const;
  bool regist(const std::string& name);
  bool regist(const std::string& name, const std::string& level_des);
  bool regist(const std::string& name, const ModuleLogConfig& config);
  bool cmd(const std::vector<std::string>& cmd, std::ostream& out);
  bool cmd_help(std::ostream& out) const;
  void flush_all();

 private:
  std::shared_ptr<GroupLoggerBase> create_logger(const std::string& name,
                                                 LogLevel level);

 private:
  LogModuleStdout _stdout_module;
  LogModuleFile _file_module;

  std::mutex _modules_mtx;
  std::unique_ptr<ModuleMapImpl> _module_map_impl;

  std::map<std::string, std::string> _cmd_des;
  LogLevel _default_level{LogLevel::INFO};
  std::string _default_level_des{"info"};  // temporary nouse

  DECLARE_SINGLETON(LogModule);
};

}  // namespace loginterface
}  // namespace framework
}  // namespace netaos

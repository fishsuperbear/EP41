#ifndef CYBER_MAINBOARD_MODULE_ARGUMENT_H_
#define CYBER_MAINBOARD_MODULE_ARGUMENT_H_

#include <list>
#include <string>

#include "framework/common/global_data.h"
#include "framework/common/log.h"
#include "framework/common/types.h"

namespace netaos {
namespace framework {
namespace mainboard {

static const char DEFAULT_process_group_[] = "mainboard_default";
static const char DEFAULT_sched_name_[] = "CYBER_DEFAULT";

class ModuleArgument {
 public:
  ModuleArgument() = default;
  virtual ~ModuleArgument() = default;
  void DisplayUsage();
  void ParseArgument(int argc, char* const argv[]);
  void GetOptions(const int argc, char* const argv[]);
  const std::string& GetBinaryName() const;
  const std::string& GetProcessGroup() const;
  const std::string& GetSchedName() const;
  const std::list<std::string>& GetDAGConfList() const;

 private:
  std::list<std::string> dag_conf_list_;
  std::string binary_name_;
  std::string process_group_;
  std::string sched_name_;
};

inline const std::string& ModuleArgument::GetBinaryName() const {
  return binary_name_;
}

inline const std::string& ModuleArgument::GetProcessGroup() const {
  return process_group_;
}

inline const std::string& ModuleArgument::GetSchedName() const {
  return sched_name_;
}

inline const std::list<std::string>& ModuleArgument::GetDAGConfList() const {
  return dag_conf_list_;
}

}  // namespace mainboard
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_MAINBOARD_MODULE_ARGUMENT_H_

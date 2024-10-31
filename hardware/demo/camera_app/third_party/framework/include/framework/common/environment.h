#ifndef CYBER_COMMON_ENVIRONMENT_H_
#define CYBER_COMMON_ENVIRONMENT_H_

#include <cassert>
#include <string>

#include "framework/common/log.h"

namespace netaos {
namespace framework {
namespace common {

inline std::string GetEnv(const std::string& var_name,
                          const std::string& default_value = "") {
  const char* var = std::getenv(var_name.c_str());
  if (var == nullptr) {
    AWARN << "Environment variable [" << var_name << "] not set, fallback to "
          << default_value;
    return default_value;
  }
  return std::string(var);
}

inline const std::string WorkRoot() {
  std::string work_root = GetEnv("NETAOS_PATH");
  if (work_root.empty()) {
    // 拿到德赛的板子后确定，暂时设置为系统根目录
    work_root = "/";
  }
  return work_root;
}

}  // namespace common
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_COMMON_ENVIRONMENT_H_

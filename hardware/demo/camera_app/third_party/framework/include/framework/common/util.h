#ifndef CYBER_COMMON_UTIL_H_
#define CYBER_COMMON_UTIL_H_

#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace netaos {
namespace framework {
namespace common {

inline std::size_t Hash(const std::string& key) {
  return std::hash<std::string>{}(key);
}

template <typename Enum>
auto ToInt(Enum const value) -> typename std::underlying_type<Enum>::type {
  return static_cast<typename std::underlying_type<Enum>::type>(value);
}

static void GetStatsLines(const std::string& stat_file, const int line_count,
                   std::vector<std::string>& stats_lines) {
  stats_lines.clear();
  std::ifstream buffer(stat_file);
  for (int line_num = 0; line_num < line_count; ++line_num) {
    std::string line;
    std::getline(buffer, line);
    if (line.empty()) {
      break;
    }
    stats_lines.push_back(line);
  }
  return;
}

}  // namespace common
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_COMMON_UTIL_H_

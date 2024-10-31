#ifndef CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_PARAM_H_
#define CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_PARAM_H_

#include <cstdint>
#include <limits>
#include <set>
#include <string>

namespace netaos {
namespace framework {
namespace record {

struct PlayParam {
  bool is_play_all_channels = false;
  bool is_loop_playback = false;
  double play_rate = 1.0;
  uint64_t begin_time_ns = 0;
  uint64_t end_time_ns = std::numeric_limits<uint64_t>::max();
  uint64_t start_time_s = 0;
  uint64_t delay_time_s = 0;
  uint32_t preload_time_s = 3;
  std::set<std::string> files_to_play;
  std::set<std::string> channels_to_play;
  std::set<std::string> black_channels;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_PARAM_H_

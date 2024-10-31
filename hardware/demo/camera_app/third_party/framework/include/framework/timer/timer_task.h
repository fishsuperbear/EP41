#ifndef CYBER_TIMER_TIMER_TASK_H_
#define CYBER_TIMER_TIMER_TASK_H_

#include <functional>
#include <mutex>

namespace netaos {
namespace framework {

class TimerBucket;

struct TimerTask {
  explicit TimerTask(uint64_t timer_id) : timer_id_(timer_id) {}
  uint64_t timer_id_ = 0;
  std::function<void()> callback;
  uint64_t interval_ms = 0;
  uint64_t remainder_interval_ms = 0;
  uint64_t next_fire_duration_ms = 0;
  int64_t accumulated_error_ns = 0;
  uint64_t last_execute_time_ns = 0;
  std::mutex mutex;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TIMER_TIMER_TASK_H_

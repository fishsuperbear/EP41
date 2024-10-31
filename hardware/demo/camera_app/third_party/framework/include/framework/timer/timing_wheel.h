#ifndef CYBER_TIMER_TIMING_WHEEL_H_
#define CYBER_TIMER_TIMING_WHEEL_H_

#include <future>
#include <list>
#include <memory>
#include <thread>

#include "framework/common/log.h"
#include "framework/common/macros.h"
#include "framework/scheduler/scheduler_factory.h"
#include "framework/time/rate.h"
#include "framework/timer/timer_bucket.h"

namespace netaos {
namespace framework {

struct TimerTask;

static const uint64_t WORK_WHEEL_SIZE = 512;
static const uint64_t ASSISTANT_WHEEL_SIZE = 64;
static const uint64_t TIMER_RESOLUTION_MS = 2;
static const uint64_t TIMER_MAX_INTERVAL_MS =
    WORK_WHEEL_SIZE * ASSISTANT_WHEEL_SIZE * TIMER_RESOLUTION_MS;

class TimingWheel {
 public:
  ~TimingWheel() {
    if (running_) {
      Shutdown();
    }
  }

  void Start();

  void Shutdown();

  void Tick();

  void AddTask(const std::shared_ptr<TimerTask>& task);

  void AddTask(const std::shared_ptr<TimerTask>& task,
               const uint64_t current_work_wheel_index);

  void Cascade(const uint64_t assistant_wheel_index);

  void TickFunc();

  inline uint64_t TickCount() const { return tick_count_; }

 private:
  inline uint64_t GetWorkWheelIndex(const uint64_t index) {
    return index & (WORK_WHEEL_SIZE - 1);
  }
  inline uint64_t GetAssistantWheelIndex(const uint64_t index) {
    return index & (ASSISTANT_WHEEL_SIZE - 1);
  }

  bool running_ = false;
  uint64_t tick_count_ = 0;
  std::mutex running_mutex_;
  TimerBucket work_wheel_[WORK_WHEEL_SIZE];
  TimerBucket assistant_wheel_[ASSISTANT_WHEEL_SIZE];
  uint64_t current_work_wheel_index_ = 0;
  std::mutex current_work_wheel_index_mutex_;
  uint64_t current_assistant_wheel_index_ = 0;
  std::mutex current_assistant_wheel_index_mutex_;
  std::thread tick_thread_;

  DECLARE_SINGLETON(TimingWheel)
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TIMER_TIMING_WHEEL_H_

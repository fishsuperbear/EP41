#ifndef CYBER_TIMER_TIMER_BUCKET_H_
#define CYBER_TIMER_TIMER_BUCKET_H_

#include <list>
#include <memory>
#include <mutex>

#include "framework/timer/timer_task.h"

namespace netaos {
namespace framework {

class TimerBucket {
 public:
  void AddTask(const std::shared_ptr<TimerTask>& task) {
    std::lock_guard<std::mutex> lock(mutex_);
    task_list_.push_back(task);
  }

  std::mutex& mutex() { return mutex_; }
  std::list<std::weak_ptr<TimerTask>>& task_list() { return task_list_; }

 private:
  std::mutex mutex_;
  std::list<std::weak_ptr<TimerTask>> task_list_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TIMER_TIMER_BUCKET_H_

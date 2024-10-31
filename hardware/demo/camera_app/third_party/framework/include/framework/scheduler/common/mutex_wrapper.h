#ifndef CYBER_SCHEDULER_COMMON_MUTEX_WRAPPER_H_
#define CYBER_SCHEDULER_COMMON_MUTEX_WRAPPER_H_

#include <mutex>

namespace netaos {
namespace framework {
namespace scheduler {

class MutexWrapper {
 public:
  MutexWrapper& operator=(const MutexWrapper& other) = delete;
  std::mutex& Mutex() { return mutex_; }

 private:
  mutable std::mutex mutex_;
};

}  // namespace scheduler
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SCHEDULER_COMMON_MUTEX_WRAPPER_H_

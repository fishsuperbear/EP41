#ifndef CYBER_BASE_THREAD_SAFE_QUEUE_H_
#define CYBER_BASE_THREAD_SAFE_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

namespace netaos {
namespace framework {
namespace base {

template <typename T>
class ThreadSafeQueue {
 public:
  ThreadSafeQueue() {}
  ThreadSafeQueue& operator=(const ThreadSafeQueue& other) = delete;
  ThreadSafeQueue(const ThreadSafeQueue& other) = delete;

  ~ThreadSafeQueue() { BreakAllWait(); }

  void Enqueue(const T& element) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace(element);
    cv_.notify_one();
  }

  bool Dequeue(T* element) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    *element = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool WaitDequeue(T* element) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return break_all_wait_ || !queue_.empty(); });
    if (break_all_wait_) {
      return false;
    }
    *element = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  typename std::queue<T>::size_type Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool Empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void BreakAllWait() {
    break_all_wait_ = true;
    cv_.notify_all();
  }

 private:
  volatile bool break_all_wait_ = false;
  std::mutex mutex_;
  std::queue<T> queue_;
  std::condition_variable cv_;
};

}  // namespace base
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_BASE_THREAD_SAFE_QUEUE_H_

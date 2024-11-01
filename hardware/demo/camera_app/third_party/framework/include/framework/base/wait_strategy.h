#ifndef CYBER_BASE_WAIT_STRATEGY_H_
#define CYBER_BASE_WAIT_STRATEGY_H_

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace netaos {
namespace framework {
namespace base {

class WaitStrategy {
 public:
  virtual void NotifyOne() {}
  virtual void BreakAllWait() {}
  virtual bool EmptyWait() = 0;
  virtual ~WaitStrategy() {}
};

class BlockWaitStrategy : public WaitStrategy {
 public:
  BlockWaitStrategy() {}
  void NotifyOne() override { cv_.notify_one(); }

  bool EmptyWait() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock);
    return true;
  }

  void BreakAllWait() override { cv_.notify_all(); }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
};

class SleepWaitStrategy : public WaitStrategy {
 public:
  SleepWaitStrategy() {}
  explicit SleepWaitStrategy(uint64_t sleep_time_us)
      : sleep_time_us_(sleep_time_us) {}

  bool EmptyWait() override {
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us_));
    return true;
  }

  void SetSleepTimeMicroSeconds(uint64_t sleep_time_us) {
    sleep_time_us_ = sleep_time_us;
  }

 private:
  uint64_t sleep_time_us_ = 10000;
};

class YieldWaitStrategy : public WaitStrategy {
 public:
  YieldWaitStrategy() {}
  bool EmptyWait() override {
    std::this_thread::yield();
    return true;
  }
};

class BusySpinWaitStrategy : public WaitStrategy {
 public:
  BusySpinWaitStrategy() {}
  bool EmptyWait() override { return true; }
};

class TimeoutBlockWaitStrategy : public WaitStrategy {
 public:
  TimeoutBlockWaitStrategy() {}
  explicit TimeoutBlockWaitStrategy(uint64_t timeout)
      : time_out_(std::chrono::milliseconds(timeout)) {}

  void NotifyOne() override { cv_.notify_one(); }

  bool EmptyWait() override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cv_.wait_for(lock, time_out_) == std::cv_status::timeout) {
      return false;
    }
    return true;
  }

  void BreakAllWait() override { cv_.notify_all(); }

  void SetTimeout(uint64_t timeout) {
    time_out_ = std::chrono::milliseconds(timeout);
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::chrono::milliseconds time_out_;
};

}  // namespace base
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_BASE_WAIT_STRATEGY_H_

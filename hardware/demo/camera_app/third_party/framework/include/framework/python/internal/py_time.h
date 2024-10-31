#ifndef CYBER_PYTHON_INTERNAL_PY_TIME_H_
#define CYBER_PYTHON_INTERNAL_PY_TIME_H_

#include <unistd.h>
#include <memory>

#include "framework/cyber.h"
#include "framework/init.h"
#include "framework/time/rate.h"
#include "framework/time/time.h"

namespace netaos {
namespace framework {

class PyTime {
 public:
  PyTime() = default;
  explicit PyTime(uint64_t nanoseconds) { time_ = Time(nanoseconds); }

  static PyTime now() {
    PyTime t;
    t.time_ = Time::Now();
    return t;
  }

  static PyTime mono_time() {
    PyTime t;
    t.time_ = Time::MonoTime();
    return t;
  }

  static void sleep_until(uint64_t nanoseconds) {
    Time::SleepUntil(Time(nanoseconds));
  }

  double to_sec() const { return time_.ToSecond(); }

  uint64_t to_nsec() const { return time_.ToNanosecond(); }

 private:
  Time time_;
};

class PyDuration {
 public:
  explicit PyDuration(int64_t nanoseconds) {
    duration_ = std::make_shared<Duration>(nanoseconds);
  }

  void sleep() const { return duration_->Sleep(); }

 private:
  std::shared_ptr<Duration> duration_ = nullptr;
};

class PyRate {
 public:
  explicit PyRate(uint64_t nanoseconds) {
    rate_ = std::make_shared<Rate>(nanoseconds);
  }

  void sleep() const { return rate_->Sleep(); }
  void reset() const { return rate_->Reset(); }
  uint64_t get_cycle_time() const { return rate_->CycleTime().ToNanosecond(); }
  uint64_t get_expected_cycle_time() const {
    return rate_->ExpectedCycleTime().ToNanosecond();
  }

 private:
  std::shared_ptr<Rate> rate_ = nullptr;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_PYTHON_INTERNAL_PY_TIME_H_

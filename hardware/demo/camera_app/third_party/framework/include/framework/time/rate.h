#ifndef CYBER_TIME_RATE_H_
#define CYBER_TIME_RATE_H_

#include "framework/time/duration.h"
#include "framework/time/time.h"

namespace netaos {
namespace framework {

class Rate {
 public:
  explicit Rate(double frequency);
  explicit Rate(uint64_t nanoseconds);
  explicit Rate(const Duration&);
  void Sleep();
  void Reset();
  Duration CycleTime() const;
  Duration ExpectedCycleTime() const { return expected_cycle_time_; }

 private:
  Time start_;
  Duration expected_cycle_time_;
  Duration actual_cycle_time_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TIME_RATE_H_

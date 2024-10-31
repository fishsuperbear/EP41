#ifndef CYBER_PYTHON_INTERNAL_PY_TIMER_H_
#define CYBER_PYTHON_INTERNAL_PY_TIMER_H_

#include <unistd.h>
#include <functional>
#include <memory>

#include "framework/cyber.h"
#include "framework/init.h"
#include "framework/timer/timer.h"

namespace netaos {
namespace framework {

class PyTimer {
 public:
  PyTimer() { timer_ = std::make_shared<Timer>(); }

  PyTimer(uint32_t period, void (*func)(), bool oneshot) {
    std::function<void()> bound_f = std::bind(func);
    timer_ = std::make_shared<Timer>(period, bound_f, oneshot);
  }

  void start() { timer_->Start(); }

  void stop() { timer_->Stop(); }

  void set_option(uint32_t period, void (*func)(), bool oneshot) {
    std::function<void()> bound_f = std::bind(func);
    TimerOption time_opt;
    time_opt.period = period;
    time_opt.callback = bound_f;
    time_opt.oneshot = oneshot;
    timer_->SetTimerOption(time_opt);
  }

 private:
  std::shared_ptr<Timer> timer_ = nullptr;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_PYTHON_INTERNAL_PY_TIMER_H_

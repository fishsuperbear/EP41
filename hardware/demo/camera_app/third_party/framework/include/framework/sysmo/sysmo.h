#ifndef CYBER_SYSMO_SYSMO_H_
#define CYBER_SYSMO_SYSMO_H_

#include <chrono>
#include <condition_variable>
#include <list>
#include <mutex>
#include <string>
#include <thread>

#include "framework/scheduler/scheduler_factory.h"

namespace netaos {
namespace framework {

using netaos::framework::scheduler::Scheduler;

class SysMo {
 public:
  void Start();
  void Shutdown();

 private:
  void Checker();

  std::atomic<bool> shut_down_{false};
  bool start_ = false;

  int sysmo_interval_ms_ = 100;
  std::condition_variable cv_;
  std::mutex lk_;
  std::thread sysmo_;

  DECLARE_SINGLETON(SysMo);
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SYSMO_SYSMO_H_

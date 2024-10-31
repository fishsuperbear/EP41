#ifndef CYBER_SCHEDULER_COMMON_PIN_THREAD_H_
#define CYBER_SCHEDULER_COMMON_PIN_THREAD_H_

#include <string>
#include <thread>
#include <vector>

#include "framework/common/log.h"

namespace netaos {
namespace framework {
namespace scheduler {

void ParseCpuset(const std::string& str, std::vector<int>* cpuset);

void SetSchedAffinity(std::thread* thread, const std::vector<int>& cpus,
                      const std::string& affinity, int cpu_id = -1);

void SetSchedPolicy(std::thread* thread, std::string spolicy,
                    int sched_priority, pid_t tid = -1);

}  // namespace scheduler
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SCHEDULER_COMMON_PIN_THREAD_H_

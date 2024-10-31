#ifndef CYBER_SCHEDULER_POLICY_SCHEDULER_CHOREOGRAPHY_H_
#define CYBER_SCHEDULER_POLICY_SCHEDULER_CHOREOGRAPHY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/croutine/croutine.h"
#include "framework/proto/choreography_conf.pb.h"
#include "framework/scheduler/scheduler.h"

namespace netaos {
namespace framework {
namespace scheduler {

using netaos::framework::croutine::CRoutine;
using netaos::framework::proto::ChoreographyTask;

class SchedulerChoreography : public Scheduler {
 public:
  bool RemoveCRoutine(uint64_t crid) override;
  bool RemoveTask(const std::string& name) override;
  bool DispatchTask(const std::shared_ptr<CRoutine>&) override;

 private:
  friend Scheduler* Instance();
  SchedulerChoreography();

  void CreateProcessor();
  bool NotifyProcessor(uint64_t crid) override;

  std::unordered_map<std::string, ChoreographyTask> cr_confs_;

  int32_t choreography_processor_prio_;
  int32_t pool_processor_prio_;

  std::string choreography_affinity_;
  std::string pool_affinity_;

  std::string choreography_processor_policy_;
  std::string pool_processor_policy_;

  std::vector<int> choreography_cpuset_;
  std::vector<int> pool_cpuset_;
};

}  // namespace scheduler
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SCHEDULER_POLICY_SCHEDULER_CHOREOGRAPHY_H_

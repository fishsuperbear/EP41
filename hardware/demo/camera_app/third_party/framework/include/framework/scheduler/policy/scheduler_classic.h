#ifndef CYBER_SCHEDULER_POLICY_SCHEDULER_CLASSIC_H_
#define CYBER_SCHEDULER_POLICY_SCHEDULER_CLASSIC_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/croutine/croutine.h"
#include "framework/proto/classic_conf.pb.h"
#include "framework/scheduler/scheduler.h"

namespace netaos {
namespace framework {
namespace scheduler {

using netaos::framework::croutine::CRoutine;
using netaos::framework::proto::ClassicConf;
using netaos::framework::proto::ClassicTask;

class SchedulerClassic : public Scheduler {
 public:
  bool RemoveCRoutine(uint64_t crid) override;
  bool RemoveTask(const std::string& name) override;
  bool DispatchTask(const std::shared_ptr<CRoutine>&) override;

 private:
  friend Scheduler* Instance();
  SchedulerClassic();

  void CreateProcessor();
  bool NotifyProcessor(uint64_t crid) override;

  std::unordered_map<std::string, ClassicTask> cr_confs_;

  ClassicConf classic_conf_;
};

}  // namespace scheduler
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SCHEDULER_POLICY_SCHEDULER_CLASSIC_H_

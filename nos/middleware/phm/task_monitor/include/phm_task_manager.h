
#ifndef PHM_TASK_MANAGER_H
#define PHM_TASK_MANAGER_H

#include <vector>
#include <map>
#include <unordered_map>
#include <memory>

#include "phm/common/include/phm_config.h"
#include "phm/include/phm_def.h"

namespace hozon {
namespace netaos {
namespace phm {

class AliveMonitor;
class DeadlineMonitor;
class LogicMonitor;
class FaultReporter;
class PHMTaskManager {
public:
    PHMTaskManager(std::shared_ptr<ModuleConfig> cfg, std::weak_ptr<FaultReporter> wpFaultReporter);
    ~PHMTaskManager();

    int32_t StartAllTask(std::vector<phm_task_t>& phm_tasks, uint32_t delayTime);
    void StopAllTask();
    int32_t ReportCheckPoint(const uint32_t checkPointId);

private:
    PHMTaskManager(const PHMTaskManager &);
    PHMTaskManager & operator = (const PHMTaskManager &);

    void AliveTimeoutHook(std::uint32_t checkpointId, bool status);
    void DeadlineTimeoutHook(phm_transition transition, bool status);
    void LogicFaultHook(uint32_t faultKey, bool status);

    bool start_;

    std::unordered_map<std::uint32_t, std::shared_ptr<AliveMonitor>> aliveTaskMap_;
    std::map<phm_transition, std::shared_ptr<DeadlineMonitor>> deadlineTaskMap_;
    std::unordered_map<std::uint32_t, std::uint8_t> aliveStatusMap_;
    std::unordered_map<std::uint32_t, std::uint8_t> deadlineStatusMap_;
    std::shared_ptr<LogicMonitor> logic_monitor_ptr_;
    std::shared_ptr<ModuleConfig> cfg_;
    std::weak_ptr<FaultReporter> m_wpFaultReporter;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_TASK_MANAGER_H

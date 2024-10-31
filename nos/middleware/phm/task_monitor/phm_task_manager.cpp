#include "phm/task_monitor/include/phm_task_manager.h"
#include "phm/task_monitor/include/phm_alive.h"
#include "phm/task_monitor/include/phm_deadline.h"
#include "phm/task_monitor/include/phm_logic.h"
#include "phm/common/include/phm_logger.h"
#include "phm/fault_manager/include/fault_reporter.h"


namespace hozon {
namespace netaos {
namespace phm {

using namespace std::placeholders;

PHMTaskManager::PHMTaskManager(std::shared_ptr<ModuleConfig> cfg, std::weak_ptr<FaultReporter> wpFaultReporter)
: start_(false)
, logic_monitor_ptr_(std::make_shared<LogicMonitor>(cfg))
, cfg_(cfg)
, m_wpFaultReporter(wpFaultReporter)
{
    PHM_DEBUG << "PHMTaskManager::PHMTaskManager";
}

PHMTaskManager::~PHMTaskManager()
{
    PHM_DEBUG << "PHMTaskManager::~PHMTaskManager";
}

int32_t
PHMTaskManager::StartAllTask(std::vector<phm_task_t>& phm_tasks, uint32_t delayTime)
{
    PHM_DEBUG << "PHMTaskManager::StartAllTask start phm_tasks.size: " << phm_tasks.size() << " delayTime: " << delayTime;
    if (start_) {
        PHM_DEBUG << "PHMTaskManager::StartAllTask already start";
        return -1;
    }

    if (0 == phm_tasks.size()) {
        return -1;
    }

    start_ = true;
    // create all task
    for (auto &task : phm_tasks) {
        if (task.monitorType == PHM_MONITOR_TYPE_ALIVE) {
            if (task.parameter[0] == 0 || task.parameter[1] == 0) {
                continue;
            }

            uint32_t minIndication = (task.parameter[1] - task.parameter[2] <= 0) ? 1 : task.parameter[1] - task.parameter[2];
            uint32_t maxIndication = task.parameter[1] + task.parameter[3];
            std::shared_ptr<AliveMonitor> aliveMonitor = AliveMonitor::MakeAliveMonitor(task.checkPointId[0], task.parameter[0], minIndication, maxIndication);
            aliveMonitor->RegistTimeoutCallbak(std::bind(&PHMTaskManager::AliveTimeoutHook, this, _1, _2));

            PHM_DEBUG << "PHMClientImpl::StartAllTask ALIVE id: " << task.checkPointId[0]
                      << ", minIndication:" << minIndication
                      << ", maxIndication:" << maxIndication;
            aliveTaskMap_[task.checkPointId[0]] = aliveMonitor;
        }
        else if (task.monitorType == PHM_MONITOR_TYPE_DEADLINE) {
            if (task.parameter[1] == 0 || task.parameter[0] > task.parameter[1]) {
                continue;
            }

            phm_transition transition {task.checkPointId[0], task.checkPointId[1]};
            std::shared_ptr<DeadlineMonitor> deadlineMonitor = DeadlineMonitor::MakeDeadlineMonitor(transition, task.parameter[0], task.parameter[1]);
            deadlineMonitor->RegistTimeoutCallbak(std::bind(&PHMTaskManager::DeadlineTimeoutHook, this, _1, _2));

            PHM_DEBUG << "PHMClientImpl::StartAllTask DEADLINE id1: " << task.checkPointId[0]
                      << " ,id2: " << task.checkPointId[1];
            deadlineTaskMap_[transition] = deadlineMonitor;
        }
        else if (task.monitorType == PHM_MONITOR_TYPE_LOGIC) {
            PHM_DEBUG << "PHMClientImpl::StartAllTask LOGIC";
        }
    }

    // start alive in delayTime seconds
    for (auto& aliveMonitorItem : aliveTaskMap_) {
        aliveMonitorItem.second->DelayedStart(delayTime);
    }

    if (logic_monitor_ptr_ != nullptr) {
        logic_monitor_ptr_->InitLogicMonitor(std::bind(&PHMTaskManager::LogicFaultHook, this, _1, _2));
    }

    PHM_DEBUG << "PHMTaskManager::StartAllTask Standby. aliveTaskMap_ size:" << aliveTaskMap_.size()
              << " deadlineTaskMap_.size:" << deadlineTaskMap_.size();
    return 0;
}

void
PHMTaskManager::StopAllTask()
{
    PHM_DEBUG << "PHMTaskManager::StopAllTask" ;
    for (auto& item : aliveTaskMap_) {
        item.second->Stop();
    }

    for (auto& item : deadlineTaskMap_) {
        item.second->Stop();
    }

    aliveTaskMap_.clear();
    deadlineTaskMap_.clear();
    aliveStatusMap_.clear();
    deadlineStatusMap_.clear();
    start_ = false;
    PHM_DEBUG << "PHMTaskManager::StopAllTask end";
}

void
PHMTaskManager::AliveTimeoutHook(std::uint32_t checkpointId, bool status)
{
    if (aliveStatusMap_[checkpointId] == 0 && status) {
        PHM_DEBUG << "PHMTaskManager::AliveTimeoutHook checkpointId: " << checkpointId << ", status: " << status;
        aliveStatusMap_[checkpointId] = 1;
        std::vector<phm_task_t>& phmTask = cfg_->GetPhmTask();
        for (auto& item : phmTask) {
            if (item.monitorType == PHM_MONITOR_TYPE_ALIVE && item.checkPointId[0] == checkpointId) {
                SendFault_t faultInfo(item.faultId, item.faultObj, 1);
                std::shared_ptr<FaultReporter> spFaultReporter = m_wpFaultReporter.lock();
                spFaultReporter->ReportFault(faultInfo, cfg_);
                break;
            }
        }
    }
    else if (aliveStatusMap_[checkpointId] == 1 && !status) {
        PHM_DEBUG << "PHMTaskManager::AliveTimeoutHook checkpointId: " << checkpointId << ", status: " << status;
        aliveStatusMap_[checkpointId] = 0;
        std::vector<phm_task_t>& phmTask = cfg_->GetPhmTask();
        for (auto& item : phmTask) {
            if (item.monitorType == PHM_MONITOR_TYPE_ALIVE && item.checkPointId[0] == checkpointId) {
                SendFault_t faultInfo(item.faultId, item.faultObj, 0);
                std::shared_ptr<FaultReporter> spFaultReporter = m_wpFaultReporter.lock();
                spFaultReporter->ReportFault(faultInfo, cfg_);
                break;
            }
        }
    }
}

void
PHMTaskManager::DeadlineTimeoutHook(phm_transition transition, bool status)
{
    PHM_DEBUG << "PHMTaskManager::DeadlineTimeoutHook checkpointSrcId: " << transition.checkpointSrcId
              << ",checkpointDestId: " << transition.checkpointDestId << ",status:" << status;
    std::vector<phm_task_t>& phmTask = cfg_->GetPhmTask();
    size_t index = 0;
    for (; index < phmTask.size(); ++index) {
        auto item = phmTask[index];
        if (item.monitorType != PHM_MONITOR_TYPE_DEADLINE) {
            continue;
        }

        if (transition.checkpointSrcId == item.checkPointId[0]
            || transition.checkpointDestId == item.checkPointId[0]) {
            break;
        }
    }

    if (phmTask.size() == index) {
        return;
    }

    uint8_t cFaultStatus = 0;
    if (deadlineStatusMap_[transition.checkpointSrcId] == 0 && status) {
        deadlineStatusMap_[transition.checkpointSrcId] = 1;
        cFaultStatus = 1;
    }
    else if (deadlineStatusMap_[transition.checkpointSrcId] == 1 && !status) {
        deadlineStatusMap_[transition.checkpointSrcId] = 0;
        cFaultStatus = 0;
    }
    else {
        return;
    }

    auto item = phmTask[index];
    SendFault_t faultInfo(item.faultId, item.faultObj, cFaultStatus);
    std::shared_ptr<FaultReporter> spFaultReporter = m_wpFaultReporter.lock();
    spFaultReporter->ReportFault(faultInfo, cfg_);
    return;
}

void
PHMTaskManager::LogicFaultHook(uint32_t faultKey, bool status)
{
    PHM_DEBUG << "PHMTaskManager::LogicTimeoutHook faultKey:" << faultKey << ",status:" << status;
    uint8_t cFaultStatus = 0;
    if (status) {
        cFaultStatus = 1;
    }

    SendFault_t faultInfo(faultKey / 100, faultKey % 100, cFaultStatus);
    std::shared_ptr<FaultReporter> spFaultReporter = m_wpFaultReporter.lock();
    spFaultReporter->ReportFault(faultInfo, cfg_);
    return;
}

int32_t
PHMTaskManager::ReportCheckPoint(const uint32_t checkPointId)
{
    PHM_DEBUG << "PHMTaskManager::ReportCheckPoint checkpointId: " << (int)checkPointId;
    if (!start_) {
        PHM_DEBUG << "PHMTaskManager::ReportCheckPoint not start";
        return -1;
    }

    if (aliveTaskMap_.count(checkPointId) != 0) {
        std::shared_ptr<AliveMonitor> aliveMonitor = aliveTaskMap_[checkPointId];
        aliveMonitor->Run();
        return 0;
    }

    for (auto &deadline : deadlineTaskMap_) {
        if (deadline.first.checkpointSrcId == checkPointId || deadline.first.checkpointDestId == checkPointId) {
            deadline.second->Run();
            return 0;
        }
    }

    if (logic_monitor_ptr_ != nullptr) {
        logic_monitor_ptr_->Run(checkPointId);
    }

    return 0;
}



}  // namespace phm
}  // namespace netaos
}  // namespace hozon
/* EOF */

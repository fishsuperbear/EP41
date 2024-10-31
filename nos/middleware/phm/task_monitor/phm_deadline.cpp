/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: deadline task monitor
*/
#include "phm/common/include/phm_logger.h"

#include "phm/task_monitor/include/phm_deadline.h"
#include "phm/common/include/timer_manager.h"

namespace hozon {
namespace netaos {
namespace phm {

using namespace std::placeholders;

std::shared_ptr<DeadlineMonitor>
DeadlineMonitor::MakeDeadlineMonitor(phm_transition transition, uint32_t deadlineMinMs, uint32_t deadlineMaxMs)
{
    std::shared_ptr<DeadlineMonitor> deadlineMonitor(
        new DeadlineMonitor(transition, deadlineMinMs, deadlineMaxMs));

    return deadlineMonitor;
}

DeadlineMonitor::DeadlineMonitor(phm_transition transition, uint32_t deadlineMinMs, uint32_t deadlineMaxMs)
: transition_(transition), start_(false)
{
    start_.store(false);

    minTask_.deadlineMs.store(deadlineMinMs);
    minTask_.timerFd.store(-1);
    minTask_.isStop.store(false);

    maxTask_.deadlineMs.store(deadlineMaxMs);
    maxTask_.timerFd.store(-1);
    maxTask_.isStop.store(false);
}

DeadlineMonitor::~DeadlineMonitor()
{
    PHM_DEBUG << "DeadlineMonitor::~DeadlineMonitor" ;
    this->Stop();
}

void DeadlineMonitor::MinTimerCallback(void* data)
{
    minTask_.isStop.store(true);
    minTask_.timerFd.store(-1);
}

void DeadlineMonitor::MaxTimerCallback(void* data)
{
    maxTask_.isStop.store(true);
    maxTask_.timerFd.store(-1);

    if (deadlineHookFunc_ != nullptr) {
        /*
        PHM_LOG_WARN << "DeadlineMonitor Exception --> cpId is " << transition_.checkpointSrcId \
            << " and " << transition_.checkpointDestId;
        */
        deadlineHookFunc_(transition_, true);
    }
}

void DeadlineMonitor::RegistTimeoutCallbak(std::function<void(phm_transition transition, bool)> deadlineHookFunc)
{
    deadlineHookFunc_ = deadlineHookFunc;
}

int DeadlineMonitor::Run()
{
    if (!start_.load()) {
        start_.store(true);
        minTask_.isStop.store(false);
        maxTask_.isStop.store(false);

        if (minTask_.deadlineMs > 0) {
            int minTaskTimerFd = minTask_.timerFd.load();
            TimerManager::Instance()->StartFdTimer(minTaskTimerFd, minTask_.deadlineMs, std::bind(&DeadlineMonitor::MinTimerCallback, this, _1), NULL, false);
            minTask_.timerFd.store(minTaskTimerFd);
        }
        else {
            minTask_.isStop.store(true);
        }

        int maxTaskTimerFd = maxTask_.timerFd.load();
        TimerManager::Instance()->StartFdTimer(maxTaskTimerFd, maxTask_.deadlineMs, std::bind(&DeadlineMonitor::MaxTimerCallback, this, _1), NULL, false);
        maxTask_.timerFd.store(maxTaskTimerFd);
    }
    else {
        start_.store(false);

        if (!minTask_.isStop.load()) {
            Stop();
            deadlineHookFunc_(transition_, true);
        }
        else if (!maxTask_.isStop.load()) {
            StopMaxTaskTimer();
            deadlineHookFunc_(transition_, false);
        }
    }

    return 0;
}

int DeadlineMonitor::Stop()
{
    PHM_INFO << "DeadlineMonitor::Stop";
    start_.store(false);
    int res1 = StopMinTaskTimer();
    int res2 = StopMaxTaskTimer();
    return (res1 == 0 && res2 == 0) ? 0 : -1;
}

int DeadlineMonitor::StopMinTaskTimer()
{
    PHM_INFO << "DeadlineMonitor::StopMinTaskTimer";
    int minTaskTimerFd = minTask_.timerFd.load();
    int res1 = TimerManager::Instance()->StopFdTimer(minTaskTimerFd);
    minTask_.timerFd.store(minTaskTimerFd);
    return (res1 == 0 ? 0 : -1);
}

int DeadlineMonitor::StopMaxTaskTimer()
{
    PHM_INFO << "DeadlineMonitor::StopMaxTaskTimer";
    int maxTaskTimerFd = maxTask_.timerFd.load();
    int res2 = TimerManager::Instance()->StopFdTimer(maxTaskTimerFd);
    maxTask_.timerFd.store(maxTaskTimerFd);
    return (res2 == 0 ? 0 : -1);
}


}  // namespace phm
}  // namespace netaos
}  // namespace hozon

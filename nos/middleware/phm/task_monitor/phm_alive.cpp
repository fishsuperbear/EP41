/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: alive task monitor
*/
#include "phm/common/include/phm_logger.h"

#include "phm/task_monitor/include/phm_alive.h"
#include "phm/common/include/timer_manager.h"


namespace hozon {
namespace netaos {
namespace phm {

using namespace std::placeholders;

std::shared_ptr<AliveMonitor>
AliveMonitor::MakeAliveMonitor(uint32_t checkpointId, uint32_t periodMs,
                               uint32_t minExpectedIndication, uint32_t maxExpectedIndication)
{
    std::shared_ptr<AliveMonitor> aliveMonitor(
        new AliveMonitor(checkpointId, periodMs, minExpectedIndication, maxExpectedIndication));

    return aliveMonitor;
}

AliveMonitor::AliveMonitor(uint32_t checkpointId, uint32_t periodMs, uint32_t minExpectedIndication, uint32_t maxExpectedIndication) :
        checkpointId_(checkpointId), periodMs_(periodMs), minExpectedIndication_(minExpectedIndication),
        maxExpectedIndication_(maxExpectedIndication), currentIndication_(0), timerFd_(-1)
{
    PHM_DEBUG << "AliveMonitor::AliveMonitor";
}

AliveMonitor::~AliveMonitor()
{
    PHM_DEBUG << "AliveMonitor::~AliveMonitor" ;
    this->Stop();
}

void AliveMonitor::TimerCallback(void* data)
{
    bool bStatus = false;
    uint32_t lockIndication = currentIndication_.load();
    uint32_t minExpectedIndication = minExpectedIndication_.load();
    uint32_t maxExpectedIndication = maxExpectedIndication_.load();

    if (minExpectedIndication == maxExpectedIndication) {
        if (lockIndication != minExpectedIndication) {

            /*
            PHM_LOG_WARN << "AliveMonitor Exception --> cpId: " << checkpointId_ << ", period: " << periodMs_ \
                            << ", minInd: " << minExpectedIndication << ", maxInd: " << maxExpectedIndication << ", reportInd: " << lockIndication;
            */
            bStatus = true;
            aliveHookFunc_(checkpointId_, bStatus);
        }
    }
    else {
        if (lockIndication < minExpectedIndication || lockIndication > maxExpectedIndication) {
            /*
            PHM_LOG_WARN << "AliveMonitor Exception --> cpId: " << checkpointId_ << ", period: " << periodMs_ \
                            << ", minInd: " << minExpectedIndication << ", maxInd: " << maxExpectedIndication_ << ", reportInd: " << lockIndication;
            */
            bStatus = true;
            aliveHookFunc_(checkpointId_, bStatus);
        }
    }

    currentIndication_.store(0);
    if (!bStatus) {
        aliveHookFunc_(checkpointId_, bStatus);
    }
}


void AliveMonitor::RegistTimeoutCallbak(std::function<void(std::uint32_t, bool status)> aliveHookFunc)
{
    aliveHookFunc_ = aliveHookFunc;
}

void AliveMonitor::DelayedStart(const uint32_t delayTime)
{
    PHM_INFO << "AliveMonitor::DelayedStart";
    if (0x00 == delayTime) {
        Start(nullptr);
        return;
    }

    PHM_INFO << "AliveMonitor::DelayedStart !0";
    int iDelayedStartTimerFd = -1;
    TimerManager::Instance()->StartFdTimer(iDelayedStartTimerFd, delayTime,
        std::bind(&AliveMonitor::Start, this, _1), NULL, false);
    return;
}

void AliveMonitor::Start(void* data)
{
    PHM_INFO << "AliveMonitor::Start";
    if (!start_.load()) {
        PHM_INFO << "AliveMonitor::Start real";
        start_.store(true);
        TimerManager::Instance()->StartFdTimer(timerFd_, periodMs_,
            std::bind(&AliveMonitor::TimerCallback, this, _1), NULL, true);
    }
}

int AliveMonitor::Run()
{
    currentIndication_++;
    return 0;
}

int AliveMonitor::Stop()
{
    PHM_INFO << "AliveMonitor::Stop";
    start_.store(false);
    currentIndication_.store(0);
    TimerManager::Instance()->StopFdTimer(timerFd_);
    return 0;
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon

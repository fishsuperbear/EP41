/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: alive task monitor
 */

#ifndef PHM_ALIVE_H
#define PHM_ALIVE_H

#include <string>
#include <unistd.h>
#include <mutex>
#include <memory>
#include <functional>


namespace hozon {
namespace netaos {
namespace phm {


class AliveMonitor {
public:
    static std::shared_ptr<AliveMonitor> MakeAliveMonitor(uint32_t checkpointId, uint32_t periodMs,
                                                          uint32_t minExpectedIndication, uint32_t maxExpectedIndication);
    ~AliveMonitor();

    int Run();
    int Stop();
    void RegistTimeoutCallbak(std::function<void(std::uint32_t, bool status)> aliveHookFunc);
    void DelayedStart(const uint32_t delayTime);
    void Start(void* data);

private:
    AliveMonitor() = default;
    AliveMonitor(uint32_t checkpointId, uint32_t periodMs, uint32_t minExpectedIndication, uint32_t maxExpectedIndication);
    void TimerCallback(void* data);

    uint32_t checkpointId_;
    uint32_t periodMs_;
    std::atomic<uint32_t> minExpectedIndication_;
    std::atomic<uint32_t> maxExpectedIndication_;
    std::atomic<uint32_t> currentIndication_;
    std::atomic<bool> start_ {false};
    int timerFd_;
    std::mutex mtx_;
    std::function<void(std::uint32_t, bool status)> aliveHookFunc_ = nullptr;

    // bool m_isNeedRecover = false;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_ALIVE_H

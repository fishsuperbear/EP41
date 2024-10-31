/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: deadline task monitor
 */

#ifndef PHM_DEADLINE_H
#define PHM_DEADLINE_H

#include <string>
#include <unistd.h>
#include <mutex>
#include <memory>
#include <atomic>
#include <functional>

#include "phm/include/phm_def.h"
#include "phm/common/include/timer_manager.h"


namespace hozon {
namespace netaos {
namespace phm {


typedef struct DeadlineTask {
    std::atomic<uint32_t> deadlineMs;
    std::atomic<int> timerFd;
    std::atomic<bool> isStop;
} DeadlineTask_t;


class DeadlineMonitor {
public:
    static std::shared_ptr<DeadlineMonitor> MakeDeadlineMonitor(phm_transition transition, uint32_t deadlineMinMs, uint32_t deadlineMaxMs);
    ~DeadlineMonitor();

    int Run();
    int Stop();
    void RegistTimeoutCallbak(std::function<void(phm_transition transition, bool)> deadlineHookFunc);

private:
    DeadlineMonitor() = default;
    DeadlineMonitor(phm_transition transition, uint32_t deadlineMinMs, uint32_t deadlineMaxMs);

    void MinTimerCallback(void* data);
    void MaxTimerCallback(void* data);

    int StopMinTaskTimer();
    int StopMaxTaskTimer();

    phm_transition transition_;
    std::atomic<bool> start_;
    DeadlineTask_t minTask_;
    DeadlineTask_t maxTask_;
    std::mutex mtx_;
    std::function<void(phm_transition transition, bool)> deadlineHookFunc_ = nullptr;
};

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_DEADLINE_H

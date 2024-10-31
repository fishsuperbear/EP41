/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fm debounce time policy
 */

#ifndef FAULT_DEBOUNCE_TIME_H
#define FAULT_DEBOUNCE_TIME_H

#include <condition_variable>
#include <mutex>
#include "phm/fault_manager/include/fault_debounce_base.h"


namespace hozon {
namespace netaos {
namespace phm {

typedef std::function<void(void)> Fun;


class DebounceTime : public DebounceBase {
public:
    DebounceTime(TimerManager* timerM, uint32_t timeout);
    virtual ~DebounceTime();

    DebouncePolicy_t GetDebouncePolicy() override;
    void StartDebounceTimer() override;
    void StopDebounceTimer() override;
    void Act() override;
    void Clear() override;
    bool isMature() override;

    void RegistDebounceTimeoutCallback(Fun fun);

private:
    // DebounceTime() = default;
    void WaitCallThread();
    void TimerCallback(void* data);

    uint32_t timeout_;
    bool isMatured_;
    Fun debounceTimeoutCallback_;
    int timerFd_ { -1 };
    std::mutex mtx_;
    bool stopFlag_;
    std::condition_variable cv_;
    std::thread waitcall_thread_;
};


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // FAULT_DEBOUNCE_TIME_H

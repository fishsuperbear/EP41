/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fm debounce count policy
 */

#ifndef FAULT_DEBOUNCE_COUNT_H
#define FAULT_DEBOUNCE_COUNT_H

#include "phm/fault_manager/include/fault_debounce_base.h"
#include <mutex>


namespace hozon {
namespace netaos {
namespace phm {


class DebounceCount : public DebounceBase {
public:
    DebounceCount(TimerManager* timerM, uint32_t maxCount, uint32_t timeout);
    virtual ~DebounceCount();

    DebouncePolicy_t GetDebouncePolicy() override;
    void StartDebounceTimer() override;
    void StopDebounceTimer() override;
    void Act() override;
    void Clear() override;
    bool isMature() override;

private:
    // DebounceCount() = default;
    void TimerCallback(void* data);

    uint32_t currentCount_;
    uint32_t maxCount_;
    uint32_t timeout_;
    int timerFd_;
    std::mutex mtx_;
};


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // FAULT_DEBOUNCE_COUNT_H

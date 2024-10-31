/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fm debounce base
 */

#ifndef FAULT_DEBOUNCE_BASE_H
#define FAULT_DEBOUNCE_BASE_H

#include <stdint.h>
#include "phm/common/include/timer_manager.h"


namespace hozon {
namespace netaos {
namespace phm {


typedef enum PolicyType {
    TYPE_COUNT = 0,
    TYPE_TIME,
} PolicyType_t;

typedef struct DebouncePolicy {
    PolicyType_t type;
    // if type is TYPE_COUNT, val[0]:maxCount, val[1]:timeout
    // if type is TYPE_TIME, val[0]:timeout, val[1]:0
    uint32_t val[2];
} DebouncePolicy_t;


class DebounceBase {
public:
    DebounceBase(TimerManager* timerM) : timerM_(timerM) {}
    virtual ~DebounceBase() {};

    virtual DebouncePolicy_t GetDebouncePolicy() = 0;
    virtual void StartDebounceTimer() = 0;
    virtual void StopDebounceTimer() = 0;
    virtual void Act() = 0;
    virtual void Clear() = 0;
    virtual bool isMature() = 0;

protected:
    DebouncePolicy_t policy_;
    TimerManager* timerM_;
};


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // FAULT_DEBOUNCE_BASE_H

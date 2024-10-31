/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fm debounce count policy
*/

#include "phm/common/include/phm_logger.h"
#include "phm/fault_manager/include/fault_debounce_count.h"

namespace hozon {
namespace netaos {
namespace phm {

using namespace std::placeholders;



DebounceCount::DebounceCount(TimerManager* timerM, uint32_t maxCount, uint32_t timeout)
    : DebounceBase(timerM), currentCount_(0), maxCount_(maxCount), timeout_(timeout), timerFd_(-1)
{
}

DebounceCount::~DebounceCount()
{
    Clear();
    StopDebounceTimer();
}

DebouncePolicy_t DebounceCount::GetDebouncePolicy()
{
    return {TYPE_COUNT, {maxCount_, timeout_}};
}

void DebounceCount::Act()
{
    std::lock_guard<std::mutex> lck(mtx_);
    ++currentCount_;
}

void DebounceCount::Clear()
{
    std::lock_guard<std::mutex> lck(mtx_);
    currentCount_ = 0;
}

void DebounceCount::StartDebounceTimer()
{
    if (-1 != timerFd_) {
        PHM_DEBUG << "DebounceCount::StartDebounceTimer already start, fd:" << timerFd_;
        return;
    }

    if (0 >= timeout_) {
        PHM_DEBUG << "DebounceCount::StartDebounceTimer timeout_:" << timeout_;
        return;
    }

    timerM_->StartFdTimer(timerFd_, timeout_, std::bind(&DebounceCount::TimerCallback, this, _1), NULL, true);
}

void DebounceCount::StopDebounceTimer()
{
    if (-1 == timerFd_) {
        return;
    }

    PHM_DEBUG << "DebounceCount::StopDebounceTimer fd:" << timerFd_;
    timerM_->StopFdTimer(timerFd_);
    timerFd_ = -1;
}

void DebounceCount::TimerCallback(void* data)
{
    std::lock_guard<std::mutex> lck(mtx_);
    currentCount_ = 0;
    timerFd_ = -1;
}

bool DebounceCount::isMature()
{
    std::lock_guard<std::mutex> lck(mtx_);
    PHM_DEBUG << "DebounceCount::isMature TYPE_COUNT currentCount_:" << currentCount_ << " maxCount_:" << maxCount_;
    return currentCount_ % maxCount_== 0 && currentCount_ != 0;
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon

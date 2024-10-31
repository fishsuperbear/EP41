/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fm debounce time policy
*/

#include "phm/fault_manager/include/fault_debounce_time.h"

namespace hozon {
namespace netaos {
namespace phm {

using namespace std::placeholders;


DebounceTime::DebounceTime(TimerManager* timerM, uint32_t timeout)
    : DebounceBase(timerM), timeout_(timeout), isMatured_(false), stopFlag_(false)
{
    waitcall_thread_ = std::thread(&DebounceTime::WaitCallThread, this);
}

DebounceTime::~DebounceTime()
{
    stopFlag_ = true;
    timerM_->StopFdTimer(timerFd_);
    cv_.notify_one();

    if (waitcall_thread_.joinable()) {
        waitcall_thread_.join();
    }
}

void DebounceTime::WaitCallThread()
{
    while (!stopFlag_) {
        std::unique_lock<std::mutex> lck(mtx_);
        cv_.wait(lck);
        if (!stopFlag_ && debounceTimeoutCallback_ != nullptr) {
            debounceTimeoutCallback_();
        }
    }
}

DebouncePolicy_t DebounceTime::GetDebouncePolicy()
{
    return {TYPE_TIME, {timeout_, 0}};
}

void DebounceTime::Act()
{
    // do nothing
}

void DebounceTime::Clear()
{
    isMatured_ = false;
}

void DebounceTime::RegistDebounceTimeoutCallback(Fun fun)
{
    debounceTimeoutCallback_ = fun;
}

void DebounceTime::StartDebounceTimer()
{
    timerM_->StartFdTimer(timerFd_, timeout_, std::bind(&DebounceTime::TimerCallback, this, _1), NULL, false);
}

void DebounceTime::StopDebounceTimer()
{
    if (-1 == timerFd_) {
        return;
    }

    timerM_->StopFdTimer(timerFd_);
}

void DebounceTime::TimerCallback(void* data)
{
    isMatured_ = true;
    timerFd_ = -1;
    cv_.notify_one();
}

bool DebounceTime::isMature()
{
    return isMatured_;
}


}  // namespace phm
}  // namespace netaos
}  // namespace hozon

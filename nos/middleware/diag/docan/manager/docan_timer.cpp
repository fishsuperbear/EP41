/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanImpl implement
 */

#include "docan_timer.h"

#include <thread>
#include <sys/time.h>
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

DocanTimerManager *DocanTimerManager::instancePtr_ = nullptr;
std::mutex DocanTimerManager::instance_mtx_;

DocanTimer::DocanTimer(uint32_t timerId, uint32_t msec, bool iter)
    : m_timerId(timerId)
    , m_timeout(msec)
    , m_isValid(true)
    , m_iter(iter)
{
}

DocanTimer::~DocanTimer()
{
    m_isValid = false;
}

int32_t
DocanTimer::StartTimer()
{
    m_isValid = true;
    m_expiredTick = GetTickCount() + m_timeout;
    DocanTimerManager::Instance()->RegisterTimer(this);
    return 0;
}

int32_t
DocanTimer::StopTimer()
{
    m_isValid = false;
    m_iter = false;
    m_timeout = 0;
    m_expiredTick = 0;
    DocanTimerManager::Instance()->UnregisterTimer(this);
    return 0;
}

int32_t
DocanTimer::RestartTimer(uint32_t new_msec)
{
    m_isValid = true;
    if (new_msec > 0 && new_msec != m_timeout) {
        m_timeout = new_msec;
    }
    m_expiredTick = GetTickCount() + m_timeout;
    DocanTimerManager::Instance()->UpdateTimer();
    return 0;
}

uint32_t
DocanTimer::GetTimeout()
{
    return m_timeout;
}

uint32_t
DocanTimer::GetExpiredTick()
{
    return m_expiredTick;
}

uint32_t
DocanTimer::GetTickCount(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

DocanTimerManager*
DocanTimerManager::Instance()
{
    if (nullptr == instancePtr_)
    {
        std::lock_guard<std::mutex> lck(instance_mtx_);
        if (nullptr == instancePtr_)
        {
            instancePtr_ = new DocanTimerManager();
        }
    }
    return instancePtr_;
}

void
DocanTimerManager::Destory()
{
    if (nullptr != instancePtr_) {
        delete instancePtr_;
        instancePtr_ = nullptr;
    }
}

DocanTimerManager::DocanTimerManager()
    : docan_timer_list_()
{
    timer_thread_ = std::thread(&DocanTimerManager::Run, this);
}

DocanTimerManager::~DocanTimerManager()
{
    for (auto it = docan_timer_list_.begin(); it != docan_timer_list_.end();) {
        DocanTimer* tmp = nullptr;
        if (*it != nullptr) {
            tmp = *it;
            it = docan_timer_list_.erase(it);
            delete tmp;
            tmp = nullptr;
        }
        else {
            ++it;
        }
    }
}

void
DocanTimerManager::notify()
{
}

void
DocanTimerManager::ThreadProc()
{
}

void
DocanTimerManager::Run()
{
}

int32_t
DocanTimerManager::Init()
{
    return 0;
}

int32_t
DocanTimerManager::Start()
{
    return 0;
}

int32_t DocanTimerManager::Stop()
{
    return 0;
}

int32_t DocanTimerManager::RegisterTimer(DocanTimer* timer)
{
    std::lock_guard<std::mutex> lck(timer_list_mutex_);
    // docan_timer_list_.push_back(timer);
    // docan_timer_list_.sort([](const DocanTimer* a, const DocanTimer* b)-> bool {
    //         return a->GetExpiredTick() > b->GetExpiredTick();
    //     })

    for (auto it = docan_timer_list_.begin(); it != docan_timer_list_.end(); ++it) {
        if ((*it)->GetExpiredTick() > timer->GetExpiredTick()) {
            docan_timer_list_.insert(it, timer);
        }
    }
    return 0;
}

int32_t DocanTimerManager::UnregisterTimer(DocanTimer* timer)
{
    std::lock_guard<std::mutex> lck(timer_list_mutex_);
    for (auto it = docan_timer_list_.begin(); it != docan_timer_list_.end();) {
        DocanTimer* tmp = nullptr;
        if (*it == timer) {
            tmp = *it;
            it = docan_timer_list_.erase(it);
            delete tmp;
            tmp = nullptr;
        }
        else {
            ++it;
        }
    }
    return 0;
}

int32_t DocanTimerManager::UpdateTimer()
{
    return 0;
}


} // end of diag
} // end of netaos
} // end of hozon
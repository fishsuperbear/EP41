/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class D Header
 */

#ifndef DOCAN_TIMER_H_
#define DOCAN_TIMER_H_

#include <list>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace hozon {
namespace netaos {
namespace diag {

class DocanTimer
{
public:
        /**
     * Construct a timer with a msec and a iter.
     *
     * @param timerid: timer id for time specifier.
     * @param msec: timer would do the action after msecs.
     * @param iter: timer iterator.
     */
    explicit DocanTimer(uint32_t timerId, uint32_t msec, bool iter = false);
    /**
     * Destructor
     */
    virtual     ~DocanTimer();

    int32_t     StartTimer();
    int32_t     StopTimer();
    int32_t     RestartTimer(uint32_t new_msec = 0);  // 0: use last timer msec

    uint32_t    GetTimeout();
    uint32_t    GetExpiredTick();

protected:
    virtual int32_t DoAction() { return 0; };
    virtual void    OnTimer() {};
    friend class DocanTimerManager;

private:
    uint32_t    GetTickCount(void);

private:
    uint32_t    m_timerId;
    uint32_t    m_timeout;
    uint32_t    m_expiredTick;
    bool        m_isValid;
    bool        m_iter;
};

class DocanTimerManager
{
public:
    static DocanTimerManager* Instance();
    static void Destory();
    virtual ~DocanTimerManager();

    int32_t Init();
    int32_t Start();
    int32_t Stop();

    int32_t RegisterTimer(DocanTimer* timer);
    int32_t UnregisterTimer(DocanTimer* timer);
    int32_t UpdateTimer();

protected:
    virtual void notify();
    virtual void ThreadProc();

private:
    DocanTimerManager(const DocanTimerManager &);
    DocanTimerManager & operator = (const DocanTimerManager &);

    DocanTimerManager();
    virtual void Run();

private:
    static DocanTimerManager* instancePtr_;
    static std::mutex instance_mtx_;

    std::condition_variable timer_condition_;
    std::thread             timer_thread_;
    std::mutex              timer_list_mutex_;
    std::list<DocanTimer*>  docan_timer_list_;


};

} // end of diag
} // end of netaos
} // end of hozon
#endif // DOIP_TIMER_H

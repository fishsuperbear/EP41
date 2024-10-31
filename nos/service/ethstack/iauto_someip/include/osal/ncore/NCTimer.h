/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file NCTimer.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTIMER_H_
#define INCLUDE_NCORE_NCTIMER_H_

#if ( defined linux ) || ( defined __linux__ )
#include <signal.h>
#include <time.h>
#endif

#include <semaphore.h>
#include <functional>
#include <memory>
#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// Class declaration
class NCTimerThreadPool;
class NCTimerRequest;
class NCTimerManager;
class NCRunnableThread;

/**
 * @brief sample code
 *
 * @code
 *
 * // must overried from NCTimer!
 * class A : public NCTimer
 * {
 *  public:
 *      A():NCTimer(timerThreadPool, 10){}
 * };
 *
 * // 2. add timers in it.
 * A mtimer;
 * mtimer.start();
 * B mbtimer;
 * mbtimer.start();
 * ...
 * mbtimer.stop();
 * mtimer.stop();
 *
 *
 */
class __attribute__( ( visibility( "default" ) ) ) NCTimer {
   public:
    /**
     * Construct a timer with a msec and a iter.
     *
     * @param msec: timer would do the action after msecs.
     * @param iter: timer iterator.
     * @param thread: this thread is work when timer is timeout and 
     *                the threads will not be released by the timer,
     *                the threads should be released by the user.
     */
    explicit NCTimer(NCRunnableThread* thread, UINT32 msec, NC_BOOL iter = NC_TRUE );

    /**
     * Construct a timer with a msec and a iter.
     *
     * @param msec: timer would do the action after msecs.
     * @param iter: timer iterator.
     */
    explicit NCTimer( UINT32 msec, NC_BOOL iter = NC_TRUE );

    /**
     * Construct a timer with a msec and a iter.
     *
     * @param msec: timer would do the action after msecs.
     * @param iter: timer iterator.
     */
    explicit NCTimer( NCTimerThreadPool *const threadPool, UINT32 msec, NC_BOOL iter = NC_TRUE );

    /**
     * Destructor
     */
    virtual ~NCTimer();

    /**
     * start the timer.
     *
     * @return  NC_TRUE indicate success, vis versa.
     */
    NC_BOOL start();

    /**
     * Stop the timer.
     *
     */
    VOID stop();

    /**
     * Restart the timer.
     *
     * @return  NC_TRUE indicate success, vis versa.
     */
    NC_BOOL restart();

    /**
     * Restart the timer after tm msecs.
     *
     * @param tm: The time that timer would restart since it's start.
     * @return  NC_TRUE indicate success, vis versa.
     */
    NC_BOOL restart( UINT32 msec );

    /**
     * Verify if timer is active.
     *
     * @return  NC_TRUE indicate acitve, vis versa.
     */
    NC_BOOL isActive() const;

    /**
     * The action that timer would do.
     */
    static VOID *doAction( VOID *arg );

    VOID runOnTimer();



    NC_BOOL isIterate() { return m_bIterate; };
    VOID    notifyAndSetOnTimerFini();
    VOID    lock();
    VOID    unlock();

   protected:
    /**
     * The function would called by doAction(). It defines
     *  actions when timer time out in this function.
     */
    virtual VOID onTimer();  // for codring rule

   private:
       /**
     * @brief Set the In Timing object when onTimer is progres
     *
     * @param intiming
     * @return VOID
     */
    VOID setInTiming( NC_BOOL intiming ) { m_intiming = intiming; }

    /**
     * @brief Get the In Timing object
     *
     * @return NC_BOOL
     */
    NC_BOOL getInTiming() const { return m_intiming; }

    NC_BOOL                         m_intiming;
    UINT32                          m_reptTime;
    NC_BOOL                         m_bIterate;
    std::shared_ptr<NCTimerRequest> m_timerequest;
    NC_BOOL                         m_isRun;
    sem_t                           m_sem;
    pthread_mutex_t                 m_lock;
    pthread_mutexattr_t             m_attr;
    NCRunnableThread*               m_thread;

   private:
    friend class NCTimerManager;

   private:
    NCTimer( const NCTimer &src );
    NCTimer &operator=( const NCTimer &src );
};

OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCTIMER_H_
/* EOF */

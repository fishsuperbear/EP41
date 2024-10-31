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
 * @file NCThread.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTHREAD_H_
#define INCLUDE_NCORE_NCTHREAD_H_

#include "osal/ncore/NCThreadSystemIF.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

// class declaration
class NCThreadBase;

/**
 * @brief
 *
 * @class NCThread
 */
class __attribute__( ( visibility( "default" ) ) ) NCThread {
   public:
    /**
     * Constructs a new thread. The thread does not begin executing
     * until startThread() is called.
     */
    NCThread();

    /**
     * Constructor with NCThreadSystemIF.
     *
     * @param NCThreadSystemIF point
     */
    explicit NCThread( NCThreadSystemIF *const sp );

    /**
     * Destructor
     */
    virtual ~NCThread();

    /**
     * Begins execution of the thread by calling run(), which should be
     * reimplemented in a NCThread subclass to contain your code.
     *
     * @param name Thread name. Default name is "Unknown Thread".
     * \note  If the thread is already running, this function does nothing.
     */
    virtual VOID startThread( const CHAR *const name = "Unknown Thread" );

    /**
     * Stop the thread.
     *
     * @param msec Default msec is INFINITE32.
     * @return NC_TRUE indicate success, vis versa.
     */
    virtual NC_BOOL stopThread( UINT32 msec = INFINITE32 );

    /**
     * Terminates the execution of the thread. The thread may or may not
     *  be terminated immediately, depending on the operating systems
     *  scheduling policies.
     *
     * @return NC_TRUE indicate success, vis versa.
     */
    virtual NC_BOOL terminate();

    /**
     * Joins a thread.
     *
     * @param msec Default msec is INFINITE32.
     * @return NC_TRUE indicate success, vis versa.
     */
    virtual NC_BOOL join( UINT32 msec = INFINITE32 );

    /**
     * Blocks the thread if this NCThread object has finished.
     * @param msec time milliseconds has elapsed. If time is INFINITE32 (the
         default), then the wait will never timeout.
     * @return NC_TRUE indicate success, vis versa.
     */
    virtual NC_BOOL waitTime( UINT32 msec = INFINITE32 );

    /**
     * Wakes one thread waiting on the wait condition.
     */
    virtual VOID notify();

    /**
     * Reset the notify status.
     */
    virtual VOID reset();

    /**
     * Check whether the thread is alive or not.
     *
     * @return NC_TRUE indicates that the thread is alive,
        otherwise the thread is not alive.
     */
    virtual NC_BOOL isAlive();

    /**
     * Check whether the thread is quit.
     *
     * @return NC_TRUE indicates that the thread is quit,
        otherwise the thread is not quit.
     */
    virtual NC_BOOL checkQuit();

    /**
     * Get the thread name.
     *
     * @return thread name.
     */
    virtual const CHAR *getName();

    /**
     * Reset thread name for a thread which already have name before.
     *
     * @param name new thread name.
     * \note It is not a reentrant function.
     *       So, do not call it in other thread.
     *       If the name argument is NULL, the name would not be changed.
     */
    virtual VOID resetName( const CHAR *const name );

    /**
     * Get the thread ID.
     *
     * @return thread ID.
     * \note If thread has not started yet, return 0.
     */
    virtual INT32 getThreadID();

    /**
     * Get the current thread priority.
     *
     * @return thread priority.
     */
    virtual INT32 getPriorityExt();

    /**
     * Set the priority for a running thread.
     *
     * @param pri new thread priority.
     * @return NC_TRUE indicate success, vis versa.
     * \note If the thread is not running, this function does nothing and
     *       returns NC_FALSE immediately.
     *       Use startThread() to start a thread first.
     */
    virtual NC_BOOL setPriorityExt( INT32 pri );
    /**
     * Set thread priority to Normal Priority.
     *
     * @return NC_TRUE indicate success, vis versa.
     * \note Normal Priority is the default priority of the operating
            system.
     */
    virtual NC_BOOL setPriorityNormal();

    /**
     * Set thread priority to Low Priority.
     *
     * @return NC_TRUE indicate success, vis versa.
     * \note Low Priority scheduled less often than Normal Priority.
     */
    virtual NC_BOOL setPriorityLow();

    /**
     * Set thread priority to High Priority.
     *
     * @return NC_TRUE indicate success, vis versa.
     * \note High Priority scheduled more often than NormalPriority.
     */
    virtual NC_BOOL setPriorityHigh();

    /**
     * This is the main part of the thread. Running the action that define
     *  in this function. The starting point for the thread. After calling
     *  startThread(), the newly created thread calls this function.
     */
    virtual VOID run();

    /**
     * Get the thread system information.
     *
     * @return NCThreadSystemIF.
     */
    virtual NCThreadSystemIF *getThreadSystem();

    /**
     * @brief Set the Thread Schedule object
     *
     * @param policy 0: SCHED_NOCHANGE; 1: SCHED_FIFO; 2: SCHED_RR; 3: SCHED_OTHER
     * @return NC_BOOL
     */
    virtual NC_BOOL setThreadSchedule( INT32 policy );

   protected:
    /**
     * Forces the current thread to sleep for secs seconds.
     *
     * \ref msleep(), usleep().
     */
    virtual VOID sleepTime( UINT64 secs );

    /**
     * Forces the current thread to sleep for msecs milliseconds.
     *
     * \ref sleep(), usleep().
     */
    virtual VOID msleepTime( UINT64 msecs );

    /**
     * Forces the current thread to sleep for usecs microseconds.
     *
     * \ref sleep(), msleep().
     */
    virtual VOID usleepTime( UINT64 usecs );

   private:
    NCThreadBase *    m_pThreadImpl;
    NCThreadSystemIF *m_pThreadSystem;

   private:
    NCThread( const NCThread & );
    NCThread &operator=( const NCThread & );
};
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCTHREAD_H_
/* EOF */

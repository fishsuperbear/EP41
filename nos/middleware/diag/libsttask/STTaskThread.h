/**
 * Copyright @ 2021 - 2023 Hozon Auto Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * Hozon Auto Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * @file  STTaskThread.h
 * @brief Class of STTaskThread
 */
#ifndef STTASKTHREAD_H
#define STTASKTHREAD_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <map>
#include <list>
#include <string>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "STObjectDef.h"
#include "STNormalTask.h"

namespace hozon {
namespace netaos {
namespace sttask {

    class STTask;
    class STStepTask;
    class STCommandTask;
    class STTimerTask;
    class STEvent;
    class STCall;
    class STTaskRunner;
    /**
     * @brief Class of STTaskThread
     *
     * This class is a normal task.
     */
    class STTaskThread
    {
    public:
        STTaskThread();  // thread deep is 10
        virtual ~STTaskThread();

        void            setTaskRunner(STTaskRunner* taskRunner);
        STTaskRunner*   getTaskRunner() const;
        void            stopAndClear();
        void            startThread(const std::string& threadName = "taskthread");
        bool            isAlive();

        uint32_t        postCommandTask(STCommandTask* cmdTask);
        uint32_t        postTimerTask(STTimerTask* timerTask);
        uint32_t        postNormalTask(STNormalTask* task);
        void            postEvent(STEvent* event);

        void            setMonitorCB(STObject* target, STObject::MonitorCB monitorCB);
        void            unsetMonitorCB(STObject* target);

        uint32_t        startEventWatcher(STObject* target, STObject::EventWatcherCB watcherCB, uint32_t timeout = ST_TIME_INFINITE);
        void            stopEventWatcher(uint32_t evtWatcherId);
        bool            isEventWatcherAlive(uint32_t evtWatcherId);

        uint32_t        startPeriod(STObject* target, STObject::PeriodCB periodCB, uint32_t periodTime);
        void            stopPeriod(uint32_t periodId);
        bool            isPeriodAlive(uint32_t periodId);

        uint32_t        scheduleEvent(const uint32_t timeout);
        void            unscheduleEvent(const uint32_t id);

        uint32_t        nextId();

    private:
        uint32_t        onNormalTaskPostResult(STNormalTask* task, uint32_t result);
        uint32_t        onStepTaskPostResult(STStepTask* stepTask, uint32_t result);

        bool            processTasks();
        void            processStepTaskResults();
        bool            processCommandTasks();
        bool            processNormalTasks();
        bool            processTopTasks();
        STNormalTask*   popTopTask();

        uint32_t        processTimerReqs();

        uint32_t        onProcessEvent(STEvent* event);
        void            onEventMonitors(STEvent* event);
        bool            onEventPeriods(STEvent* event);
        bool            onEventWatchers(STEvent* event);
        bool            onEventStepTasks(STEvent* event);
        void            onUnexpectedEventMonitors(STEvent* event);

        void            run();
        uint32_t        GetTickCount();

    private:
        STTaskThread(const STTaskThread&);
        STTaskThread& operator=(const STTaskThread&);

        STTaskRunner*               m_taskRunner;
        uint32_t                    m_nextId;

        std::mutex                  m_syncObj;
        std::condition_variable     m_condVar;

        typedef std::list<STNormalTask*>  TASK_QUEUE;
        typedef std::list<STStepTask*>    STEP_QUEUE;
        STEP_QUEUE                  m_stepQueue;
        TASK_QUEUE                  m_executeQueue;
        TASK_QUEUE                  m_topTaskQueue;

        std::thread                 m_thread;
        bool                        m_isStop;

        typedef std::map<STObject*, STObject::MonitorCB> MONITOR_CB_MAP;
        MONITOR_CB_MAP              m_eventMonitorCBs;

        /**
         * @brief struct STTimerReq
         *
         * This struct is STTimerReq.
         */
        struct STTimerReq
        {
            uint32_t id;
            uint32_t scheduleTime;
            uint32_t timeout;
        };

        typedef std::list<STTimerReq*>  TIMERREQ_QUEUE;

        TIMERREQ_QUEUE             m_timerReqQueue;

        /**
         * @brief struct STEventWatcher
         *
         * This struct is STEventWatcher.
         */
        struct STEventWatcher
        {
            uint32_t                    id;
            uint32_t                    timeout;
            uint32_t                    timerEventId;
            STObject*                   target;
            STObject::EventWatcherCB    watcherCB;
        };

        typedef std::list<STEventWatcher*>  EVENTWATCHER_QUEUE;

        EVENTWATCHER_QUEUE              m_evtWatcherReqQueue;

        /**
         * @brief struct STPeriod
         *
         * This struct is STPeriod.
         */
        struct STPeriod
        {
            uint32_t                    id;
            uint32_t                    periodTime;
            uint32_t                    timerEventId;
            uint32_t                    periodNo;
            STObject*                   target;
            STObject::PeriodCB          periodCB;
        };

        typedef std::list<STPeriod*>  PERIOD_QUEUE;

        PERIOD_QUEUE                    m_periodQueue;



        /**
         * @brief class STTaskProcessEvent
         *
         * This class is task event process.
         */
        class STTaskProcessEvent: public STNormalTask
        {
        public:
            explicit STTaskProcessEvent(STEvent* event);
            virtual ~STTaskProcessEvent();

            virtual std::string         toString();

        protected:
            virtual uint32_t            doAction();

        private:
            STTaskProcessEvent(const STTaskProcessEvent&);
            STTaskProcessEvent& operator=(const STTaskProcessEvent&);
            STEvent*                    m_event;
        };

    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STTASKTHREAD_H */
/* EOF */
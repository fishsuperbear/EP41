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
 * @file STTaskThread.cpp
 * @brief implements of STTaskThread
 */

#include "STTaskThread.h"
#include "STLogDef.h"
#include "STTask.h"
#include "STCommandTask.h"
#include "STTimerTask.h"
#include "STEvent.h"
#include "STCall.h"
#include "STTaskConfig.h"
#include "STTaskContext.h"
#include "STModuleManager.h"
#include "STTaskRunner.h"

#define NO_WAIT ((uint32_t)0)

namespace hozon {
namespace netaos {
namespace sttask {

    STTaskThread::STTaskThread()
        : m_taskRunner(nullptr)
        , m_nextId(0)
        , m_isStop(false)
    {
    }

    STTaskThread::~STTaskThread()
    {
        m_isStop = true;
        stopAndClear();
    }


    void STTaskThread::setTaskRunner(STTaskRunner* taskRunner)
    {
        m_taskRunner = taskRunner;
    }

    STTaskRunner* STTaskThread::getTaskRunner() const
    {
        return m_taskRunner;
    }

    void STTaskThread::run()
    {
        m_thread = std::thread([this](){
            while (!m_isStop) {
                if (processTasks()) {
                    continue;
                }

                // wait(wait_time);
                uint32_t wait_time = processTimerReqs();
                if (NO_WAIT == wait_time) {
                    continue;
                }
                std::unique_lock<std::mutex> autoSync(m_syncObj);
                // STLOG_D("wait wait_time: %d ms", wait_time);
                wait_time = wait_time < 50 ? wait_time : 50;
                m_condVar.wait_for(autoSync, std::chrono::milliseconds(wait_time));
                // STLOG_D("wake up");
            }
        });
    }

    void STTaskThread::startThread(const std::string& threadName)
    {
        m_isStop = false;
        run();
        pthread_setname_np(m_thread.native_handle(), threadName.c_str());
    }

    bool STTaskThread::isAlive()
    {
        return !m_isStop;
    }

    void STTaskThread::stopAndClear()
    {
        // 1. stop thread
        m_isStop = true;
        m_condVar.notify_all();
        if (m_thread.joinable()) {
            m_thread.join();
        }

        // 2. clear queue
        m_timerReqQueue.clear();
        m_topTaskQueue.clear();
        m_executeQueue.clear();
        m_stepQueue.clear();
        m_eventMonitorCBs.clear();
        m_evtWatcherReqQueue.clear();
    }

    uint32_t STTaskThread::postCommandTask(STCommandTask* cmdTask)
    {
        if (nullptr == cmdTask
            || nullptr == getTaskRunner()) {
            STLOG_E("Memery error");
            return eMemErr;
        }

        uint32_t result = getTaskRunner()->checkStepTaskPost(cmdTask);
        result = onStepTaskPostResult(cmdTask, result);
        if (eContinue != result) {
            return result;
        }
        cmdTask->onPost(getTaskRunner());
        m_stepQueue.push_back(cmdTask);
        return eContinue;
    }

    uint32_t STTaskThread::postTimerTask(STTimerTask* timerTask)
    {
        if (nullptr == timerTask
            || nullptr == getTaskRunner()
            || nullptr == getTaskRunner()->getContext()) {
            STLOG_E("Memery error");
            return eMemErr;
        }

        uint32_t result = getTaskRunner()->checkStepTaskPost(timerTask);
        result = onStepTaskPostResult(timerTask, result);
        if (eContinue != result) {
            return result;
        }

        timerTask->onPost(getTaskRunner());
        if (getTaskRunner()->checkStepTaskExecute(timerTask)) {
            result = timerTask->doTask();
            if (eContinue == result) {
                m_stepQueue.push_back(timerTask);

            }
            return result;
        }

        return eError;
    }

    uint32_t STTaskThread::postNormalTask(STNormalTask* task)
    {
        if (nullptr == task
            || nullptr == getTaskRunner()) {
            STLOG_E("Memery error");
            return eError;
        }

        bool isTopTask = task->isTopTask();
        if (isTopTask) {
            std::lock_guard<std::mutex> autoSync(m_syncObj);
            uint32_t result = getTaskRunner()->checkNormalTaskPost(task);
            result = onNormalTaskPostResult(task, result);
            if (eContinue != result) {
                return result;
            }
            task->onPost(getTaskRunner());
            m_topTaskQueue.push_back(task);
            m_condVar.notify_all();
            // STLOG_D("Top task to notify thread continue task...");
        }
        else {
            uint32_t result = getTaskRunner()->checkNormalTaskPost(task);
            result = onNormalTaskPostResult(task, result);
            if (eContinue != result) {
                return result;
            }
            task->onPost(getTaskRunner());
            m_executeQueue.push_back(task);
        }
        return eContinue;
    }

    void STTaskThread::postEvent(STEvent* event)
    {
        if (event) {
            postNormalTask(new STTaskProcessEvent(event));
        }
    }

    void STTaskThread::setMonitorCB(STObject* target, STObject::MonitorCB monitorCB)
    {
        if (target && monitorCB) {
            m_eventMonitorCBs[target] = monitorCB;
        }
    }

    void STTaskThread::unsetMonitorCB(STObject* target)
    {
        if (target && m_eventMonitorCBs.count(target)) {
            m_eventMonitorCBs.erase(target);
        }
    }

    uint32_t STTaskThread::startEventWatcher(STObject* target, STObject::EventWatcherCB watcherCB, uint32_t timeout)
    {
        if (nullptr == target
            || nullptr == watcherCB
            || 0 == timeout) {
            return 0;
        }

        STEventWatcher* evtWatcher = new STEventWatcher;
        if (nullptr == evtWatcher) {
            return 0;
        }
        memset(evtWatcher, 0, sizeof(STEventWatcher));
        evtWatcher->id = nextId();
        evtWatcher->target = target;
        evtWatcher->watcherCB = watcherCB;

        evtWatcher->timeout = timeout;
        if (timeout != ST_TIME_INFINITE) {
            evtWatcher->timerEventId = scheduleEvent(timeout);
        }
        m_evtWatcherReqQueue.push_back(evtWatcher);
        return evtWatcher->id;
    }

    void STTaskThread::stopEventWatcher(uint32_t evtWatcherId)
    {
        if (0 == evtWatcherId) {
            return;
        }

        for (EVENTWATCHER_QUEUE::iterator it = m_evtWatcherReqQueue.begin(); it != m_evtWatcherReqQueue.end();) {
            STEventWatcher* tmp = *it;
            if (evtWatcherId == tmp->id) {
                it = m_evtWatcherReqQueue.erase(it);
                unscheduleEvent(tmp->timerEventId);
                delete tmp;
                tmp = nullptr;
                break;
            }
            else {
                ++it;
            }
        }
    }

    bool STTaskThread::isEventWatcherAlive(uint32_t evtWatcherId)
    {
        if (0 == evtWatcherId) {
            return false;
        }

        for (auto it : m_evtWatcherReqQueue) {
            if (evtWatcherId == it->id) {
                return true;
            }
        }
        return false;
    }


    uint32_t STTaskThread::startPeriod(STObject* target, STObject::PeriodCB periodCB, uint32_t periodTime)
    {
        if (nullptr == target
            || nullptr == periodCB
            || 0 == periodTime
            || ST_TIME_INFINITE == periodTime) {
            return 0;
        }

        STPeriod* period = new STPeriod;
        if (nullptr == period) {
            return 0;
        }
        memset(period, 0, sizeof(STPeriod));
        period->id = nextId();
        period->periodTime = periodTime;
        period->target = target;
        period->periodCB = periodCB;


        ((*period->target).*period->periodCB)(period->id, period->periodNo);
        period->periodNo += 1;

        period->timerEventId = scheduleEvent(period->periodTime);

        m_periodQueue.push_back(period);
        return period->id;

    }

    void STTaskThread::stopPeriod(uint32_t periodId)
    {
        if (0 == periodId) {
            return;
        }

        for (PERIOD_QUEUE::iterator it = m_periodQueue.begin(); it != m_periodQueue.end();) {
            STPeriod* tmp = *it;
            if (periodId == tmp->id) {
                it = m_periodQueue.erase(it);
                unscheduleEvent(tmp->timerEventId);
                delete tmp;
                break;
            }
            else {
                ++it;
            }
        }
    }

    bool STTaskThread::isPeriodAlive(uint32_t periodId)
    {
        if (0 == periodId) {
            return false;
        }

        for (auto it : m_periodQueue) {
            if (periodId == it->id) {
                return true;
            }
        }
        return false;
    }

    uint32_t STTaskThread::scheduleEvent(const uint32_t timeout)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        STTimerReq* tmrReq = new STTimerReq;
        uint32_t tick = GetTickCount();
        tmrReq->scheduleTime = tick + timeout;
        tmrReq->id = nextId();
        tmrReq->timeout = timeout;

        if (m_timerReqQueue.size()) {
            bool bInsert = false;
            for (auto it = m_timerReqQueue.begin(); it != m_timerReqQueue.end(); ++it) {
                if (tmrReq->scheduleTime < (*it)->scheduleTime) {
                    m_timerReqQueue.insert(it, tmrReq);
                    bInsert = true;
                    break;
                }
            }
            if (!bInsert) {
                // need push back in the end.
                m_timerReqQueue.push_back(tmrReq);
            }
        }
        else {
            m_timerReqQueue.push_back(tmrReq);
        }
        // STLOG_D("scheduleEvent m_timerReqQueue insert timeId: %d, timeout: %d, scheduleTime: %d, current queue size: %ld.", tmrReq->id, timeout, tmrReq->scheduleTime, m_timerReqQueue.size());
        m_condVar.notify_all();
        // STLOG_D("notify task thread to continue... ");
        return tmrReq->id;
    }

    void STTaskThread::unscheduleEvent(const uint32_t id)
    {
        // STLOG_D("unscheduleEvent timeId: %d, m_timerReqQueue size: %ld.", id, m_timerReqQueue.size());
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        for (auto &it : m_timerReqQueue) {
            if (id == it->id) {
                it->id = 0; // 0 for canceled
                break;
            }
        }
    }

    uint32_t STTaskThread::nextId()
    {
        uint32_t newId = m_nextId++;
        while (0 == newId) {
            newId = m_nextId++;
        }
        return newId;
    }

    uint32_t STTaskThread::onNormalTaskPostResult(STNormalTask* task, uint32_t result)
    {
        if (task && task->isTopTask()) {
            if (eDeleteFront == result) {
                uint32_t operationId = task->getOperationId();

                for (TASK_QUEUE::iterator it = m_topTaskQueue.begin(); it != m_topTaskQueue.end();) {
                    STNormalTask* tmp = *it;
                    if (nullptr == tmp) {
                        break;
                    }
                    if (tmp->getOperationId() == operationId) {
                        it = m_topTaskQueue.erase(it);
                        tmp->deleteTask();
                        tmp = nullptr;
                        break;
                    }
                    else {
                        ++it;
                    }
                }

                return eContinue;
            }
            else if (eDeleteBack == result) {
                uint32_t operationId = task->getOperationId();
                for (TASK_QUEUE::reverse_iterator rit = m_topTaskQueue.rbegin(); rit != m_topTaskQueue.rend();) {
                    STNormalTask* tmp = *rit;
                    if (tmp->getOperationId() == operationId) {
                        rit = TASK_QUEUE::reverse_iterator(m_topTaskQueue.erase((++rit).base()));
                        tmp->deleteTask();
                        tmp = nullptr;
                        break;
                    }
                    else {
                        ++rit;
                    }
                }

                return eContinue;
            }
            else {
            }
        }
        return result;
    }

    uint32_t STTaskThread::onStepTaskPostResult(STStepTask* stepTask, uint32_t result)
    {
        (void)(stepTask);
        return result;
    }

    bool STTaskThread::processTasks()
    {
        processStepTaskResults();
        if (processCommandTasks()) {
            return true;
        }

        if (processNormalTasks()) {
            return true;
        }

        if (processTopTasks()) {
            return true;
        }

        return false;
    }

    void STTaskThread::processStepTaskResults()
    {
        STEP_QUEUE finished;
        {
            for (STEP_QUEUE::iterator it = m_stepQueue.begin(); it != m_stepQueue.end();) {
                STStepTask* stepTask = *it;
                if (nullptr == stepTask) {
                    break;
                }
                uint32_t taskResult = stepTask->getTaskResult();
                if (eNone != taskResult) {
                    it = m_stepQueue.erase(it);
                    finished.push_back(stepTask);
                }
                else {
                    ++it;
                }
            }
        }

        {
            for (STEP_QUEUE::iterator it = finished.begin(); it != finished.end(); ++it) {
                STStepTask* stepTask = *it;
                if (nullptr == stepTask) {
                    break;
                }
                uint32_t taskResult = stepTask->getTaskResult();
                stepTask->onCallbackResult(taskResult);
            }
        }
    }

    bool STTaskThread::processCommandTasks()
    {
        bool taskExecuted = false;
        for (STEP_QUEUE::iterator it = m_stepQueue.begin(); it != m_stepQueue.end(); ++it) {
            STStepTask* stepTask = *it;
            if (nullptr == stepTask
                || nullptr == getTaskRunner()) {
                break;
            }

            if (!stepTask->isCommandTask()) {
                continue;
            }

            STCommandTask* cmdTask = CAST_COMMANDTASK(stepTask);
            if (nullptr != cmdTask && !cmdTask->isExecuted()) {
                if (getTaskRunner()->checkStepTaskExecute(cmdTask)) {
                    uint32_t taskResult = cmdTask->doTask();
                    if (eContinue != taskResult) {
                        cmdTask->setTaskResult(taskResult);
                        taskExecuted = true;
                    }
                    else {
                        // must be status of wait, if eContinue returned
                        if (!cmdTask->isWaitEvent()) {
                            cmdTask->setTaskResult(eError);
                            taskExecuted = true;
                        }
                    }
                }
            }
        }
        return taskExecuted;
    }

    bool STTaskThread::processNormalTasks()
    {
        bool taskExecuted = false;
        for (TASK_QUEUE::iterator it = m_executeQueue.begin(); it != m_executeQueue.end();) {
            STNormalTask* task = *it;
            if (nullptr == task) {
                break;
            }
            it = m_executeQueue.erase(it);
            taskExecuted = true;

            uint32_t taskResult = task->getTaskResult();
            if (eNone != taskResult) {
                task->onCallbackResult(taskResult);
            }
            else {
                taskResult = task->doTask();
                if (eContinue != taskResult) {
                    task->onCallbackResult(taskResult);
                }
            }
        }
        return taskExecuted;
    }

    bool STTaskThread::processTopTasks()
    {
        STNormalTask* task = popTopTask();

        if (nullptr == task) {
            return false;
        }

        uint32_t taskResult = task->getTaskResult();
        if (eNone != taskResult) {
            task->onCallbackResult(taskResult);
        }
        else {
            taskResult = task->doTask();
            if (eContinue != taskResult) {
                task->onCallbackResult(taskResult);
            }
        }

        return true;
    }

    STNormalTask* STTaskThread::popTopTask()
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        for (TASK_QUEUE::iterator it = m_topTaskQueue.begin(); it != m_topTaskQueue.end();) {
            STNormalTask* task = *it;
            if (nullptr == task
                || nullptr == getTaskRunner()) {
                break;
            }
            if (getTaskRunner()->checkOperationExecute(task)) {
                it = m_topTaskQueue.erase(it);
                return task;
            }
            else {
                ++it;
            }
        }

        return nullptr;
    }

    uint32_t STTaskThread::processTimerReqs()
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        uint32_t cur_time = GetTickCount();
        uint32_t firedId = 0;
        for (TIMERREQ_QUEUE::iterator it = m_timerReqQueue.begin(); it != m_timerReqQueue.end();) {
            STTimerReq* tmp = *it;
            if (nullptr == tmp) {
                break;
            }
            if (0 == tmp->id) {
                it = m_timerReqQueue.erase(it);
                delete tmp;
                tmp = nullptr;
            }
            else {
                if ((tmp->scheduleTime > cur_time) && (cur_time > tmp->timeout || tmp->scheduleTime < tmp->timeout)) {
                    return (tmp->scheduleTime - cur_time);
                }
                if ((tmp->scheduleTime < cur_time) && (tmp->scheduleTime < tmp->timeout && cur_time > tmp->timeout)
                    && (cur_time + tmp->timeout) < tmp->scheduleTime) {
                    return ((0xFFFFFFFF - cur_time) + tmp->scheduleTime);
                }

                firedId = tmp->id;
                it = m_timerReqQueue.erase(it);
                delete tmp;
                tmp = nullptr;
                break;
            }
        }

        m_syncObj.unlock();
        if (firedId != 0) {
            STLOG_D("need post timeout event, firedId: %d.", firedId);
            postEvent(new STEvent(eEventKind_TimerEvent, firedId));
            return NO_WAIT;
        }
        // return INFINITE current for 6 min;
        return 6*60*1000;

    }

    uint32_t STTaskThread::onProcessEvent(STEvent* event)
    {
        // 1. monitor process evt
        onEventMonitors(event);

        // 2. period process evt(timer event)
        if (onEventPeriods(event)) {
            return eOK;
        }

        // 3. watcher process evt
        if (onEventWatchers(event)) {
            return eOK;
        }

        // 4. step tasks process evt
        if (onEventStepTasks(event)) {
            return eOK;
        }

        // 4. monitor process unexpected event
        onUnexpectedEventMonitors(event);
        return eOK;
    }

    void STTaskThread::onEventMonitors(STEvent* event)
    {
        if (nullptr == event) {
            return;
        }

        if (getTaskRunner()) {
            getTaskRunner()->onMonitorEvent(event);
        }

        for (MONITOR_CB_MAP::iterator it = m_eventMonitorCBs.begin(); it != m_eventMonitorCBs.end(); ++it) {
            STObject* target = it->first;
            STObject::MonitorCB cb = it->second;
            ((*target).*cb)(event);
        }

    }

    bool STTaskThread::onEventPeriods(STEvent* event)
    {
        if (nullptr == event) {
            return false;
        }

        if (event->getEventKind() != eEventKind_TimerEvent) {
            return false;
        }

        for (PERIOD_QUEUE::iterator it = m_periodQueue.begin(); it != m_periodQueue.end();) {
            STPeriod* tmp = *it;
            if (nullptr == tmp) {
                break;
            }

            if (event->getEventId() == tmp->timerEventId) {

                ((*tmp->target).*tmp->periodCB)(tmp->id, tmp->periodNo);
                tmp->periodNo += 1;
                tmp->timerEventId = scheduleEvent(tmp->periodTime);
                return true;
            }
        }

        return false;

    }

    bool STTaskThread::onEventWatchers(STEvent* event)
    {
        if (nullptr == event) {
            return false;
        }
        for (EVENTWATCHER_QUEUE::iterator it = m_evtWatcherReqQueue.begin(); it != m_evtWatcherReqQueue.end();) {
            STEventWatcher* tmp = *it;
            if (nullptr == tmp) {
                break;
            }

            if (STEvent::isTimerEvent(tmp->timerEventId, event)) {
                ((*tmp->target).*tmp->watcherCB)(true, event);
                it = m_evtWatcherReqQueue.erase(it);
                delete tmp;
                tmp = nullptr;
                return true;
            }
            else {
                if (((*tmp->target).*tmp->watcherCB)(false, event)) {
                    return true;
                }
                ++it;
            }
        }
        return false;
    }

    bool STTaskThread::onEventStepTasks(STEvent* event)
    {
        for (STEP_QUEUE::iterator it = m_stepQueue.begin(); it != m_stepQueue.end(); ++it) {
            STStepTask* stepTask = *it;
            if (nullptr == stepTask) {
                break;
            }

            if (stepTask->onEvent(event)) {
                return true;
            }
        }
        return false;
    }

    void STTaskThread::onUnexpectedEventMonitors(STEvent* event)
    {
        if (nullptr == event) {
            return;
        }

        if (getTaskRunner()) {
            getTaskRunner()->onMonitorUnexpectedEvent(event);
        }
    }

    uint32_t STTaskThread::GetTickCount()
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
    }

    STTaskThread::STTaskProcessEvent::STTaskProcessEvent(STEvent* event)
        : STNormalTask(eOperation_HandleEvent, nullptr, nullptr, true)
        , m_event(event)
    {
    }

    STTaskThread::STTaskProcessEvent::~STTaskProcessEvent()
    {
        if (nullptr != m_event) {
            delete m_event;
            m_event = nullptr;
        }
    }

    std::string STTaskThread::STTaskProcessEvent::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), "Operation(%p, %d, %s)"
            , this
            , getOperationId()
            , (nullptr == m_event ? "<nullptr>" : m_event->toString().c_str()));
        val.assign(buf, strlen(buf));
        return val;
    }

    uint32_t STTaskThread::STTaskProcessEvent::doAction()
    {
        if (m_event
            && getTaskRunner()
            && getTaskRunner()->getTaskThread()) {
            return getTaskRunner()->getTaskThread()->onProcessEvent(m_event);
        }

        return eError;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
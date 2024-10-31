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
 * @file STTask.cpp
 * @brief implements of STTask
 */


#include "STTask.h"

#include "STTaskRunner.h"
#include "STLogDef.h"
#include "STEvent.h"
#include "STCall.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STTask::STTask(ST_TASK_TYPE taskType, uint32_t operationId, STObject* parent, STObject::TaskCB callback, bool isTopTask)
        : STObject(ST_OBJECT_TYPE_TASK)
        , m_taskType(taskType)
        , m_operationId(operationId)
        , m_isTopTask(isTopTask)
        , m_parent(parent)
        , m_taskCallback(callback)
        , m_taskResult(eNone)
        , m_interruptReason(ST_TASK_INTERRUPTREASON_INVALID)
    {
        STTask* topTask = getTopTask();
        if (topTask) {
            m_operationId = topTask->getOperationId();
        }
    }

    STTask::~STTask()
    {
    }

    ST_TASK_TYPE STTask::getTaskType() const
    {
        return m_taskType;
    }

    uint32_t STTask::getOperationId() const
    {
        return m_operationId;
    }

    bool STTask::isTopTask() const
    {
        return m_isTopTask;
    }

    bool STTask::isNormalTask() const
    {
        return ST_TASK_TYPE_NORMAL == getTaskType() ? true : false;
    }

    bool STTask::isTimerTask() const
    {
        return ST_TASK_TYPE_TIMER == getTaskType() ? true : false;
    }

    bool STTask::isCommandTask() const
    {
        return ST_TASK_TYPE_COMMAND == getTaskType() ? true : false;
    }

    bool STTask::isStepTask() const
    {
        return isCommandTask() || isTimerTask();
    }


    uint32_t STTask::getTaskResult() const
    {
        return m_taskResult;
    }

    void STTask::setTaskResult(uint32_t result)
    {
        if (eNone == result) {
            STLOG_W("set task result eNone(%d).", result);
            return;
        }

        if (eContinue == result) {
            STLOG_W("set task result eContinue(%d).", result);
            return;
        }

        if (eNone != m_taskResult) {
            return;
        }
        m_taskResult = result;
    }

    void STTask::onCallbackResult(uint32_t result)
    {
        setTaskResult(result);
        onCallbackAction(result);
        onTaskEnd(result);
        if (m_parent && m_taskCallback) {
            ((*m_parent).*m_taskCallback)(this, result);
        }
        deleteTask();
    }

    void STTask::onPost(STTaskRunner* runner)
    {
        setTaskRunner(runner);
        onTaskPost();
    }

    bool STTask::isIntterrupted() const
    {
        return (ST_TASK_INTERRUPTREASON_INVALID == m_interruptReason) ? false : true;
    }

    uint32_t STTask::getIntteruptedReason() const
    {
        return m_interruptReason;
    }

    uint32_t STTask::post(STTask* task)
    {
        if (nullptr != task) {
            if (getTaskRunner()) {
                return getTaskRunner()->post(task);
            }
            delete task;
            task = nullptr;
        }

        return eMemErr;
    }

    void STTask::post(STEvent* event)
    {
        if (nullptr != event) {
            if (getTaskRunner()) {
                getTaskRunner()->post(event);
                return;
            }
            delete event;
            event  = nullptr;
        }
    }

    void STTask::call(STCall* callObj)
    {
        if (nullptr != callObj) {
            if (getTaskRunner()) {
                getTaskRunner()->call(callObj);
                return;
            }
            delete callObj;
            callObj = nullptr;
        }
    }

    void STTask::setMonitorCB(STObject* target, STObject::MonitorCB monitorCB)
    {
        if (getTaskRunner()) {
            getTaskRunner()->setMonitorCB(target, monitorCB);
            return;
        }
    }

    void STTask::unsetMonitorCB(STObject* target)
    {
        if (getTaskRunner()) {
            getTaskRunner()->unsetMonitorCB(target);
            return;
        }
    }

    uint32_t STTask::startEventWatcher(STObject* target, STObject::EventWatcherCB watcherCB, uint32_t timeout)
    {
        if (getTaskRunner()) {
            return getTaskRunner()->startEventWatcher(target, watcherCB, timeout);
        }
        return 0;
    }

    void STTask::stopEventWatcher(uint32_t evtWatcherId)
    {
        if (getTaskRunner()) {
            getTaskRunner()->stopEventWatcher(evtWatcherId);
            return;
        }
    }

    uint32_t STTask::startPeriod(STObject* target, STObject::PeriodCB periodCB, uint32_t periodTime)
    {
        if (getTaskRunner()) {
            return getTaskRunner()->startPeriod(target, periodCB, periodTime);
        }
        return 0;
    }

    void STTask::stopPeriod(uint32_t periodId)
    {
        if (getTaskRunner()) {
            getTaskRunner()->stopPeriod(periodId);
            return;
        }
    }

    uint32_t STTask::scheduleEvent(const uint32_t timeout)
    {
        if (getTaskRunner()) {
            return getTaskRunner()->scheduleEvent(timeout);
        }
        return 0;
    }

    void STTask::unscheduleEvent(const uint32_t id)
    {
        if (getTaskRunner()) {
            getTaskRunner()->unscheduleEvent(id);
            return;
        }
    }

    STTaskContext* STTask::getContext()
    {
        if (getTaskRunner()) {
            return getTaskRunner()->getContext();
        }
        return nullptr;
    }

    STModuleManager* STTask::getModuleManager()
    {
        if (getTaskRunner()) {
            return getTaskRunner()->getModuleManager();
        }
        return nullptr;
    }

    uint32_t STTask::doTask()
    {
        uint32_t result = onTaskStart();
        if (eContinue == result) {
            result = doAction();
        }
        return result;
    }

    void STTask::deleteTask()
    {
        if (getTaskRunner()) {
            getTaskRunner()->onTaskDelete(this);
        }
        delete this;
    }

    STTask* STTask::getTopTask()
    {
        if (isTopTask()) {
            return this;
        }

        STObject* parentObj = m_parent;
        while (nullptr != parentObj && ST_OBJECT_TYPE_TASK == parentObj->getObjectType()) {
            STTask* parentTask = CAST_TASK(parentObj);
            if (nullptr != parentTask) {
                if (parentTask->isTopTask()) {
                    return parentTask;
                }

                parentObj = parentTask->m_parent;
            }
            else {
                break;
            }
        }

        return nullptr;
    }

    STObject* STTask::getParent() const
    {
        return m_parent;
    }

    std::string STTask::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), "%s[%p, %d]", getObjectName().c_str(), this, getOperationId());
        val.assign(buf, strlen(buf));
        return val;
    }

    std::string STTask::getObjectName()
    {
        return isTopTask() ? std::string("Operation") : std::string("Task");
    }

    void STTask::onInterruptAction(uint32_t interruptReason)
    {
        // implemented by child class
        (void)(interruptReason);
    }

    void STTask::onInterrupt(uint32_t interruptReason)
    {
        m_interruptReason = interruptReason;
        onInterruptAction(interruptReason);
        if (getTaskRunner()) {
            getTaskRunner()->onTaskInterrupt(this, interruptReason);
        }

        STObject* parentObj = m_parent;
        if (nullptr != parentObj && ST_OBJECT_TYPE_TASK == parentObj->getObjectType()) {
            STTask* parentTask = CAST_TASK(parentObj);
            if (nullptr != parentTask) {
                parentTask->onInterrupt(interruptReason);
            }
        }
    }

    void STTask::onTaskPost()
    {
        if (getTaskRunner()) {
            getTaskRunner()->onTaskPost(this);
        }
    }

    uint32_t STTask::onTaskStart()
    {
        if (getTaskRunner()) {
            return getTaskRunner()->onTaskStart(this);
        }
        return eError;
    }

    void STTask::onTaskEnd(uint32_t result)
    {
        if (getTaskRunner()) {
            getTaskRunner()->onTaskEnd(this, result);
        }
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
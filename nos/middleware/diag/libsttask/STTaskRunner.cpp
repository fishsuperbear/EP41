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
 * @file STTaskRunner.cpp
 * @brief implements of STTaskRunner
 */

#include "STTaskRunner.h"
#include "STLogDef.h"
#include "STCommandTask.h"
#include "STTimerTask.h"
#include "STNormalTask.h"
#include "STEvent.h"
#include "STCall.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STTaskRunner::STTaskRunner()
        : STObject(ST_OBJECT_TYPE_TASKRUNNER)
    {
        STTaskLogger::GetInstance().CreateLogger("task");
        m_taskThread.setTaskRunner(this);
        m_configuration.setTaskRunner(this);
        m_context.setTaskRunner(this);
        m_moduleManager.setTaskRunner(this);
        setTaskRunner(this);
    }

    STTaskRunner::~STTaskRunner()
    {
    }

    void STTaskRunner::init()
    {
        STTaskConfig::OPERATION_CONFIG config;
        config.maxOperationCount = ST_TASK_SIZE_UNLIMITED;
        config.queueMethod = 0;
        getConfiguration()->registerOperation(eOperation_HandleEvent, config);
        onInit();
        getContext()->init(getConfiguration());
    }

    void STTaskRunner::start()
    {
        getTaskThread()->startThread(getTaskThreadName());
        onStart();
    }

    void STTaskRunner::stop()
    {
        onStop();
        getTaskThread()->stopAndClear();
        onStopped();
    }

    void STTaskRunner::deinit()
    {
        onDeinit();
    }

    STTaskThread* STTaskRunner::getTaskThread()
    {
        return &m_taskThread;
    }

    STTaskConfig* STTaskRunner::getConfiguration()
    {
        return &m_configuration;
    }

    STTaskContext* STTaskRunner::getContext()
    {
        return &m_context;
    }

    STModuleManager* STTaskRunner::getModuleManager()
    {
        return &m_moduleManager;
    }


    uint32_t STTaskRunner::post(STTask* task)
    {
        if (nullptr == task) {
            STLOG_E("Memery error");
            return eMemErr;
        }

        if (!getTaskThread()->isAlive()) {
            STLOG_E("TaskThread is not alive.");
            task->deleteTask();
            return eError;
        }

        if (task->isCommandTask()) {
            uint32_t result = getTaskThread()->postCommandTask(CAST_COMMANDTASK(task));
            if (eContinue != result) {
                STLOG_W("Task(%s) is posted error(%d).", task->toString().c_str(), result);
                task->deleteTask();
            }
            return result;
        }
        else if (task->isTimerTask()) {
            uint32_t result = getTaskThread()->postTimerTask(CAST_TIMERTASK(task));
            if (eContinue != result) {
                STLOG_W("Task(%s) is posted error(%d).", task->toString().c_str(), result);
                task->deleteTask();
            }
            return result;
        }
        else if (task->isNormalTask()) {
            uint32_t result = getTaskThread()->postNormalTask(CAST_NORMALTASK(task));
            if (eContinue != result) {
                STLOG_W("Task(%s) is posted error(%d).", task->toString().c_str(), result);
                task->deleteTask();
            }
            return result;
        }
        else {
            STLOG_E("Unknown type of task(%s).", task->toString().c_str());
        }
        task->deleteTask();
        return eError;

    }

    void STTaskRunner::post(STEvent* event)
    {
        if (nullptr == event) {
            STLOG_E("Memery error");
            return;
        }

        if (!getTaskThread()->isAlive()) {
            STLOG_E("TaskThread is not alive.");
            delete event;
            event = nullptr;
            return;
        }

        getTaskThread()->postEvent(event);
    }

    void STTaskRunner::call(STCall* callObj)
    {
        if (callObj) {
            getModuleManager()->onCall(callObj->getCallKind(), callObj->getCallId(), callObj);
            onCall(callObj->getCallKind(), callObj->getCallId(), callObj);
            delete callObj;
            callObj = nullptr;
        }
    }

    void STTaskRunner::setMonitorCB(STObject* target, STObject::MonitorCB monitorCB)
    {
        getTaskThread()->setMonitorCB(target, monitorCB);
    }

    void STTaskRunner::unsetMonitorCB(STObject* target)
    {
        getTaskThread()->unsetMonitorCB(target);
    }

    uint32_t STTaskRunner::startEventWatcher(STObject* target, STObject::EventWatcherCB watcherCB, uint32_t timeout)
    {
        return getTaskThread()->startEventWatcher(target, watcherCB, timeout);
    }

    void STTaskRunner::stopEventWatcher(uint32_t evtWatcherId)
    {
        getTaskThread()->stopEventWatcher(evtWatcherId);
    }

    bool STTaskRunner::isEventWatcherAlive(uint32_t evtWatcherId)
    {
        return getTaskThread()->isEventWatcherAlive(evtWatcherId);
    }


    uint32_t STTaskRunner::startPeriod(STObject* target, STObject::PeriodCB periodCB, uint32_t periodTime)
    {
        return getTaskThread()->startPeriod(target, periodCB, periodTime);
    }

    void STTaskRunner::stopPeriod(uint32_t periodId)
    {
        getTaskThread()->stopPeriod(periodId);
    }

    bool STTaskRunner::isPeriodAlive(uint32_t periodId)
    {
        return getTaskThread()->isPeriodAlive(periodId);
    }

    uint32_t STTaskRunner::scheduleEvent(const uint32_t timeout)
    {
        if (!getTaskThread()->isAlive()) {
            STLOG_E("TaskThread is not alive.");
            return 0;
        }
        return getTaskThread()->scheduleEvent(timeout);
    }

    void STTaskRunner::unscheduleEvent(const uint32_t id)
    {
        getTaskThread()->unscheduleEvent(id);
    }

    uint32_t STTaskRunner::nextId()
    {
        return getTaskThread()->nextId();
    }

    bool STTaskRunner::interruptOperation(uint32_t operationId, uint32_t interruptReason)
    {
        return getContext()->interruptOperation(operationId, interruptReason);
    }

    bool STTaskRunner::interruptCommand(uint32_t commandId, uint32_t interruptReason)
    {
        return getContext()->interruptCommand(commandId, interruptReason);
    }


    void STTaskRunner::onInit()
    {
    }

    void STTaskRunner::onStart()
    {
    }

    void STTaskRunner::onStop()
    {
    }

    void STTaskRunner::onStopped()
    {
    }

    void STTaskRunner::onDeinit()
    {
    }

    void STTaskRunner::onOperationPost(uint32_t operationId, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(topTask);
    }

    uint32_t STTaskRunner::onOperationStart(uint32_t operationId, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(topTask);
        return eContinue;
    }

    void STTaskRunner::onOperationStarted(uint32_t operationId, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(topTask);
    }

    void STTaskRunner::onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(result);
        (void)(topTask);
    }

    void STTaskRunner::onOperationInterrupt(uint32_t operationId, uint32_t interruptReason, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(interruptReason);
        (void)(topTask);
    }

    void STTaskRunner::onStepPost(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(stepTask);
    }

    uint32_t STTaskRunner::onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(stepTask);
        return eContinue;
    }

    void STTaskRunner::onStepStarted(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(stepTask);
    }

    void STTaskRunner::onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(result);
        (void)(stepTask);
    }

    void STTaskRunner::onStepInterrupt(uint32_t operationId, uint32_t stepId, uint32_t interruptReason, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(interruptReason);
        (void)(stepTask);
    }

    void STTaskRunner::onEvent(uint32_t eventKind, uint32_t eventId, STEvent* event)
    {
        (void)(eventKind);
        (void)(eventId);
        (void)(event);
    }

    void STTaskRunner::onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event)
    {
        (void)(eventKind);
        (void)(eventId);
        (void)(event);
    }

    void STTaskRunner::onCall(uint32_t callKind, uint32_t callId, STCall* callObj)
    {
        (void)(callKind);
        (void)(callId);
        (void)(callObj);
    }

    void STTaskRunner::onTaskPost(STTask* task)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return;
        }

        if (task->isStepTask()) {
            STStepTask* stepTask = CAST_STEPTASK(task);
            if (nullptr != stepTask) {
                getContext()->onStepPost(stepTask);
                doOnStepPost(stepTask);
            }
            else {
                STLOG_E("Memory error.");
            }
        }
        else if (task->isNormalTask()) {
            STNormalTask* opTask = CAST_NORMALTASK(task);
            if (nullptr != opTask) {
                if (opTask->isTopTask()) {
                    getContext()->onOperationPost(opTask);
                    doOnOperationPost(opTask);
                }
                else {
                    return;
                }
            }
            else {
                STLOG_E("Memory error.");
            }
        }
        else {
            STLOG_E("Unknown type of task(%s).", task->toString().c_str());
        }
    }

    uint32_t STTaskRunner::onTaskStart(STTask* task)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return eMemErr;
        }
        if (task->isStepTask()) {
            STStepTask* stepTask = CAST_STEPTASK(task);
            if (nullptr != stepTask) {
                uint32_t result = doOnStepStart(stepTask);
                if (eContinue == result) {
                    getContext()->onStepStart(stepTask);
                }
                else {
                    STLOG_W("Cannot do step(%s): result = %d.", stepTask->toString().c_str(), result);
                }

                return result;
            }
            else {
                STLOG_E("Memory error.");
                return eMemErr;
            }
        }
        else if (task->isNormalTask()) {
            STNormalTask* opTask = CAST_NORMALTASK(task);
            if (nullptr != opTask) {
                if (opTask->isTopTask()) {
                    uint32_t result = doOnOperationStart(opTask);
                    if (eContinue == result) {
                        getContext()->onOperationStart(opTask);
                    }
                    else {
                        STLOG_W("Cannot do operation(%s): result = %d.", opTask->toString().c_str(), result);
                    }
                    return result;
                }
                else {
                    return eContinue;
                }
            }
            else {
                STLOG_E("Memory error.");
                return eMemErr;
            }
        }
        else {
            STLOG_E("Unknown task type.");
            return eError;
        }
    }

    void STTaskRunner::onTaskEnd(STTask* task, uint32_t result)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return;
        }

        if (task->isStepTask()) {
            STStepTask* stepTask = CAST_STEPTASK(task);
            if (nullptr != stepTask) {
                getContext()->onStepEnd(stepTask, result);
                doOnStepEnd(stepTask, result);
            }
            else {
                STLOG_E("Memory error.");
            }
        }
        else if (task->isNormalTask()) {
            STNormalTask* opTask = CAST_NORMALTASK(task);
            if (nullptr != opTask) {
                if (opTask->isTopTask()) {
                    getContext()->onOperationEnd(opTask, result);
                    doOnOperationEnd(opTask, result);
                }
                else {
                    return;
                }
            }
            else {
                STLOG_E("Memory error.");
            }
        }
        else {
            STLOG_E("Unknown task type.");
        }
    }

    void STTaskRunner::onTaskDelete(STTask* task)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return;
        }

        if (task->isStepTask()) {
            STStepTask* stepTask = CAST_STEPTASK(task);
            if (nullptr != stepTask) {
                getContext()->onStepDelete(stepTask);
            }
            else {
                STLOG_E("Memory error.");
            }
        }
        else if (task->isNormalTask()) {
            STNormalTask* opTask = CAST_NORMALTASK(task);
            if (nullptr != opTask) {
                if (opTask->isTopTask()) {
                    getContext()->onOperationDelete(opTask);
                }
                else {
                    return;
                }
            }
            else {
                STLOG_E("Memory error.");
            }
        }
        else {
            STLOG_E("Unknown task type.");
        }
    }

    void STTaskRunner::onTaskInterrupt(STTask* task, uint32_t interruptReason)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return;
        }

        if (task->isStepTask()) {
            STStepTask* stepTask = CAST_STEPTASK(task);
            if (nullptr != stepTask) {
                doOnStepInterrupt(stepTask, interruptReason);
            }
            else {
                STLOG_E("Memory error.");
            }
        }
        else if (task->isNormalTask()) {
            STNormalTask* opTask = CAST_NORMALTASK(task);
            if (nullptr != opTask) {
                if (opTask->isTopTask()) {
                    doOnOperationInterrupt(opTask, interruptReason);
                }
                else {
                    return;
                }
            }
            else {
                STLOG_E("Memory error.");
            }
        }
        else {
            STLOG_E("Unknown task type.");
        }
    }

    uint32_t STTaskRunner::checkNormalTaskPost(STNormalTask* task)
    {
        if (nullptr == task) {
            STLOG_E("Memery error");
            return eMemErr;
        }

        uint32_t operationId = task->getOperationId();
        bool isTopTask = task->isTopTask();

        if (!getConfiguration()->isRegisteredOperation(operationId)) {
            STLOG_E("OperationId(%d) of Task(%s) is not registed.", operationId, task->toString().c_str());
            return eNotRegisteredOperation;
        }

        if (isTopTask) {
            uint32_t maxOperationCount = getConfiguration()->getMaxOperationCount(operationId);
            uint8_t  operationQueueMethod = getConfiguration()->getOperationQueueMethod(operationId);
            if (ST_TASK_SIZE_UNLIMITED == maxOperationCount) {
                // do nothing here.
            }
            else {
                uint32_t curOperationCount = getContext()->curOperationCount(operationId);

                bool curOperationExecuting = getContext()->isOperationExecuting(operationId);

                uint32_t curOperationCountNotExecuting = curOperationCount;
                uint32_t maxOperationCountNotExecuting = maxOperationCount;

                if (curOperationExecuting) {
                    if (curOperationCountNotExecuting > 0) {
                        curOperationCountNotExecuting -= 1;
                    }
                }

                if (getConfiguration()->testOperationQueueMethod(operationQueueMethod, STTaskConfig::QUEUE_METHOD_INCLUDE_EXECUTING)) {
                    if (curOperationExecuting) {
                        if (maxOperationCountNotExecuting > 0) {
                            maxOperationCountNotExecuting -= 1;
                        }
                    }
                }
                else {
                }


                if (curOperationCountNotExecuting < maxOperationCountNotExecuting) {

                }
                else {
                    // process if overflow
                    if (getConfiguration()->testOperationQueueMethod(operationQueueMethod, STTaskConfig::QUEUE_METHOD_DELETE_SELF_IF_OVERFLOW)) {
                        STLOG_E("Operation(%d) is overflow, so Task(%s) posted error.", operationId, task->toString().c_str());
                        return eOperationChannelOverflow;
                    }
                    else if (getConfiguration()->testOperationQueueMethod(operationQueueMethod, STTaskConfig::QUEUE_METHOD_DELETE_FRONT_IF_OVERFLOW)) {
                        if (curOperationCountNotExecuting > 0) {
                                return eDeleteFront;
                            }
                        STLOG_E("Operation(%d) is overflow, so Task(%s) posted error.", operationId, task->toString().c_str());
                        return eOperationChannelOverflow;
                    }
                    else if (getConfiguration()->testOperationQueueMethod(operationQueueMethod, STTaskConfig::QUEUE_METHOD_DELETE_BACK_IF_OVERFLOW)) {
                        if (curOperationCountNotExecuting > 0) {
                            return eDeleteBack;
                        }
                        STLOG_E("Operation(%d) is overflow, so Task(%s) posted error.", operationId, task->toString().c_str());
                        return eOperationChannelOverflow;
                    }
                    else {
                        STLOG_E("Unknown queue method, so Task(%s) posted error.", task->toString().c_str());
                        return eError;
                    }
                }

            }
        }

        return eContinue;
    }

    uint32_t STTaskRunner::checkStepTaskPost(STStepTask* stepTask)
    {
       if (nullptr == stepTask) {
            STLOG_E("Memery error");
            return eMemErr;
        }

        if (stepTask->isCommandTask()) {
            STCommandTask* cmdTask = CAST_COMMANDTASK(stepTask);
            if (nullptr == cmdTask) {
                return eMemErr;
            }

            uint32_t commandId = cmdTask->getStepId();

            if (!getConfiguration()->isRegisteredCommand(commandId)) {
                STLOG_E("CommandId(%d) of Command(%s) is not registed.", commandId, cmdTask->toString().c_str());
                return eNotRegisteredCommand;
            }

            uint32_t commandChannelId = getConfiguration()->getCommandChannel(commandId);
            uint32_t channelMaxSize = getConfiguration()->getCommandChannelMaxSize(commandChannelId);

            if (ST_TASK_SIZE_UNLIMITED == channelMaxSize) {
                // do nothing here.
            }
            else {
                uint32_t curCommandCount = getContext()->curCommandCountInChannel(commandChannelId);
                if (curCommandCount < channelMaxSize) {
                    // do nothere here
                }
                else {
                    STLOG_E("CommandChannel(%d) is overflow, so Command(%s) posted error.", commandChannelId, stepTask->toString().c_str());
                    return eCommandChannelOverflow;
                }
            }

            return eContinue;
        }
        else if (stepTask->isTimerTask()) {
            STTimerTask* timerTask = CAST_TIMERTASK(stepTask);
            if (nullptr == timerTask) {
                return eMemErr;
            }

            uint32_t timerId = timerTask->getStepId();

            if (!getConfiguration()->isRegisteredTimer(timerId)) {
                STLOG_E("TimerId(%d) of Timer(%s) is not registed.", timerId, timerTask->toString().c_str());
                return eNotRegisteredTimer;
            }

            return eContinue;
        }
        else {
            STLOG_E("invalid step task.");
        }

        return eError;
    }


    bool STTaskRunner::checkOperationExecute(STNormalTask* task)
    {
        if (nullptr == task) {
            return false;
        }
        uint32_t operationId = task->getOperationId();
        bool isTopTask = task->isTopTask();
        if (!getConfiguration()->isRegisteredOperation(operationId)) {
            STLOG_E("OperationId(%d) of Task(%s) is not registed.", operationId, task->toString().c_str());
            return false;
        }

        if (!isTopTask) {
            STLOG_E("Not a top task(operation)");
            return false;
        }

        if (getContext()->isOperationExecuting(operationId)) {
            return false;
        }

        return true;

    }

    bool STTaskRunner::checkStepTaskExecute(STStepTask* stepTask)
    {
       if (nullptr == stepTask) {
            return false;
        }

        if (stepTask->isCommandTask()) {
            STCommandTask* cmdTask = CAST_COMMANDTASK(stepTask);
            if (nullptr == cmdTask) {
                return false;
            }

            uint32_t commandId = cmdTask->getStepId();
            uint32_t commandChannelId = getConfiguration()->getCommandChannel(commandId);

            if (!getConfiguration()->isRegisteredCommand(commandId)) {
                STLOG_E("CommandId(%d) of Command(%s) is not registed.", commandId, cmdTask->toString().c_str());
                return false;
            }

            if (getContext()->isCommandExcutingInChannel(commandChannelId)) {
                STLOG_I("CommandChannel(%d) is busy, Command(%s) cannot execute.", commandChannelId, cmdTask->toString().c_str());
                return false;
            }

            return true;
        }
        else if (stepTask->isTimerTask()) {
            STTimerTask* timerTask = CAST_TIMERTASK(stepTask);
            if (nullptr == timerTask) {
                return false;
            }

            uint32_t timerId = timerTask->getStepId();

            if (!getConfiguration()->isRegisteredTimer(timerId)) {
                STLOG_E("TimerId(%d) of Timer(%s) is not registed.", timerId, timerTask->toString().c_str());
                return false;
            }
            return true;
        }
        else {
            STLOG_E("invalid step task.");
        }

        return false;

    }

    void STTaskRunner::onMonitorEvent(STEvent* event)
    {
        if (event) {
            getModuleManager()->onEvent(event->getEventKind(), event->getEventId(), event);
            onEvent(event->getEventKind(), event->getEventId(), event);
        }
    }

    void STTaskRunner::onMonitorUnexpectedEvent(STEvent* event)
    {
        if (event) {
            getModuleManager()->onUnexpectedEvent(event->getEventKind(), event->getEventId(), event);
            onUnexpectedEvent(event->getEventKind(), event->getEventId(), event);
        }
    }

    void STTaskRunner::doOnStepPost(STStepTask* stepTask)
    {
        if (stepTask) {
            getModuleManager()->onStepPost(stepTask->getOperationId(), stepTask->getStepId(), stepTask);
            onStepPost(stepTask->getOperationId(), stepTask->getStepId(), stepTask);
        }
    }

    void STTaskRunner::doOnOperationPost(STNormalTask* opTask)
    {
        if (opTask) {
            getModuleManager()->onOperationPost(opTask->getOperationId(), opTask);
            onOperationPost(opTask->getOperationId(), opTask);
        }
    }

    uint32_t STTaskRunner::doOnStepStart(STStepTask* stepTask)
    {
        uint32_t result = eContinue;
        if (stepTask) {
            result = getModuleManager()->onStepStart(stepTask->getOperationId(), stepTask->getStepId(), stepTask);

            if (eContinue == result) {
                result = onStepStart(stepTask->getOperationId(), stepTask->getStepId(), stepTask);
            }
            if (eContinue == result) {
                getModuleManager()->onStepStarted(stepTask->getOperationId(), stepTask->getStepId(), stepTask);
                onStepStarted(stepTask->getOperationId(), stepTask->getStepId(), stepTask);
            }
        }
        return result;
    }

    uint32_t STTaskRunner::doOnOperationStart(STNormalTask* opTask)
    {
        uint32_t result = eContinue;
        if (opTask) {
            result = getModuleManager()->onOperationStart(opTask->getOperationId(), opTask);

            if (eContinue == result) {
                result = onOperationStart(opTask->getOperationId(), opTask);
            }
            if (eContinue == result) {
                getModuleManager()->onOperationStarted(opTask->getOperationId(), opTask);
                onOperationStarted(opTask->getOperationId(), opTask);
            }
        }
        return result;
    }


    void STTaskRunner::doOnStepEnd(STStepTask* stepTask, uint32_t result)
    {
        if (stepTask) {
            getModuleManager()->onStepEnd(stepTask->getOperationId(), stepTask->getStepId(), result, stepTask);
            onStepEnd(stepTask->getOperationId(), stepTask->getStepId(), result, stepTask);
        }
    }

    void STTaskRunner::doOnOperationEnd(STNormalTask* opTask, uint32_t result)
    {
        if (opTask) {
            getModuleManager()->onOperationEnd(opTask->getOperationId(), result, opTask);
            onOperationEnd(opTask->getOperationId(), result, opTask);
        }
    }

    void STTaskRunner::doOnStepInterrupt(STStepTask* stepTask, uint32_t interruptReason)
    {
        if (stepTask) {
            getModuleManager()->onStepInterrupt(stepTask->getOperationId(), stepTask->getStepId(), interruptReason, stepTask);
            onStepInterrupt(stepTask->getOperationId(), stepTask->getStepId(), interruptReason, stepTask);
        }
    }

    void STTaskRunner::doOnOperationInterrupt(STNormalTask* opTask, uint32_t interruptReason)
    {
        if (opTask) {
            getModuleManager()->onOperationInterrupt(opTask->getOperationId(), interruptReason, opTask);
            onOperationInterrupt(opTask->getOperationId(), interruptReason, opTask);
        }
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */

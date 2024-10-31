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
 * @file STTaskContext.cpp
 * @brief implements of STTaskContext
 */

#include "STTaskContext.h"
#include "STLogDef.h"

#include "STTaskThread.h"
#include "STTaskRunner.h"
#include "STTask.h"
#include "STCommandTask.h"
#include "STTimerTask.h"
#include "STNormalTask.h"
#include "STEvent.h"
#include "STCall.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STTaskContext::STTaskContext()
        : STObject(ST_OBJECT_TYPE_CONTEXT)
    {
    }

    STTaskContext::~STTaskContext()
    {
    }

    void STTaskContext::init(STTaskConfig* configuration)
    {
        if (nullptr == configuration) {
            return;
        }

        for (STTaskConfig::OPERATION_CONFIG_MAP::iterator it = configuration->m_registeredOperations.begin();
            it != configuration->m_registeredOperations.end();
            ++it) {

            uint32_t operationId = it->first;

            OPERATION_TRACE newTrace;
            newTrace.curOperationTask = nullptr;
            newTrace.curStepTask = nullptr;
            m_operationTraces[operationId] = newTrace;
            STLOG_D("Create OperationTrace(%d).", operationId);
        }

        for (STTaskConfig::CHANNEL_MAP::iterator it = configuration->m_commandChannels.begin();
            it != configuration->m_commandChannels.end();
            ++it) {

            uint32_t commandChannelId = it->first;

            COMMAND_TRACE newTrace;
            newTrace.executing = nullptr;
            m_commandTraces[commandChannelId] = newTrace;
            STLOG_D("Create commandTrace(%d).", commandChannelId);
        }
    }

    bool STTaskContext::isOperationExist(uint32_t operationId)
    {
        return curOperationCount(operationId) > 0 ? true : false;
    }

    bool STTaskContext::isOperationExecuting(uint32_t operationId)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            if (nullptr != m_operationTraces[operationId].curOperationTask) {
                return true;
            }
        }

        return false;
    }

    bool STTaskContext::interruptOperation(uint32_t operationId, uint32_t interruptReason)
    {
        if (m_operationTraces.count(operationId) > 0) {
            if (nullptr != m_operationTraces[operationId].curOperationTask) {
                if (nullptr != m_operationTraces[operationId].curStepTask) {
                    return m_operationTraces[operationId].curStepTask->setInterrupted(interruptReason);
                }
            }
        }
        return false;
    }

    bool STTaskContext::restoreOperation(uint32_t operationId)
    {
        if (m_operationTraces.count(operationId) > 0) {
            if (nullptr != m_operationTraces[operationId].curOperationTask) {
                if (nullptr != m_operationTraces[operationId].curStepTask) {
                    m_operationTraces[operationId].curStepTask->setTaskResult(eNone);
                    return true;
                }
            }
        }
        return false;
    }

    bool STTaskContext::isOperationExecutingTask(STNormalTask* opTask)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (nullptr == opTask) {
            return false;
        }

        uint32_t operationId = opTask->getOperationId();
        if (m_operationTraces.count(operationId) > 0) {
            if (opTask == m_operationTraces[operationId].curOperationTask) {
                return true;
            }
        }

        return false;
    }

    uint32_t STTaskContext::curOperationCount(uint32_t operationId)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            return m_operationTraces[operationId].allOperationTasks.size();
        }
        STLOG_E("Unexpected operationId(%d).", operationId);
        return 0;
    }

    uint32_t STTaskContext::curOperationStep(uint32_t operationId)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            if (nullptr != m_operationTraces[operationId].curOperationTask) {
                if (nullptr != m_operationTraces[operationId].curStepTask) {
                    return m_operationTraces[operationId].curStepTask->getStepId();
                }
            }
        }
        return ST_TASK_INVALID_STEP;
    }

    bool STTaskContext::setCurOperationStepTask(uint32_t operationId, uint32_t result)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            if (nullptr != m_operationTraces[operationId].curOperationTask) {
                if (nullptr != m_operationTraces[operationId].curStepTask) {
                    m_operationTraces[operationId].curStepTask->setTaskResult(result);
                    return true;
                }
            }
        }
        return false;
    }

    bool STTaskContext::isOperationStepExecuting(uint32_t operationId, uint32_t stepId)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            if (nullptr != m_operationTraces[operationId].curOperationTask) {
                STStepTask* stepTask = m_operationTraces[operationId].curStepTask;
                if (nullptr != stepTask && stepTask->getStepId() == stepId) {
                    if (stepTask->isTimerTask()) {
                        return true;
                    }
                    else if (stepTask->isCommandTask()) {
                        STCommandTask* cmdTask = CAST_COMMANDTASK(stepTask);
                        if (nullptr != cmdTask) {
                            return cmdTask->isExecuted();
                        }
                    }
                    else {
                        STLOG_E("Unexpected invalid step task.");
                    }
                }
            }
        }
        return false;
    }

    uint32_t STTaskContext::curCommandCountInChannel(uint32_t commandChannelId)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_commandTraces.count(commandChannelId) > 0) {
            return m_commandTraces[commandChannelId].commandTasks.size();
        }
        STLOG_E("Unexpected commandChannelId(%d).", commandChannelId);
        return 0;
    }

    bool STTaskContext::isCommandExcutingInChannel(uint32_t commandChannelId)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_commandTraces.count(commandChannelId) > 0) {
            if (nullptr != m_commandTraces[commandChannelId].executing) {
                STLOG_I("CommandTrace(%d) is executing Command(%s)", commandChannelId, m_commandTraces[commandChannelId].executing->toString().c_str());
                return true;
            }
        }
        return false;
    }

    uint32_t STTaskContext::curExecutingCommandInChannel(uint32_t commandChannelId)
    {
        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_commandTraces.count(commandChannelId) > 0) {
            if (nullptr != m_commandTraces[commandChannelId].executing) {
                return m_commandTraces[commandChannelId].executing->getStepId();
            }
        }
        return ST_TASK_INVALID_STEP;
    }

    bool STTaskContext::interruptCommand(uint32_t commandId, uint32_t interruptReason)
    {
        if (nullptr == getTaskRunner()
            || nullptr == getTaskRunner()->getConfiguration()) {
            STLOG_E("Memory error.");
            return false;
        }

        uint32_t commandChannelId = getTaskRunner()->getConfiguration()->getCommandChannel(commandId);
        if (m_commandTraces.count(commandChannelId) > 0) {
            if (nullptr != m_commandTraces[commandChannelId].executing) {
                return m_commandTraces[commandChannelId].executing->setInterrupted(interruptReason);
            }
        }
        return false;
    }

    void STTaskContext::onStepPost(STStepTask* stepTask)
    {
        if (nullptr == stepTask
            || nullptr == getTaskRunner()
            || nullptr == getTaskRunner()->getConfiguration()) {
            STLOG_E("Memory error.");
            return;
        }

        uint32_t stepId = stepTask->getStepId();
        uint32_t operationId = stepTask->getOperationId();

        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            if (nullptr != m_operationTraces[operationId].curStepTask) {
                STLOG_E("OperationTrace(%d)'s curStepTask must be nullptr, but now is Task(%s).", operationId, m_operationTraces[operationId].curStepTask->toString().c_str());
            }
            m_operationTraces[operationId].curStepTask = stepTask;
            // STLOG_D("OperationTrace(%d) set curStepTask from nullptr to Task(%s).", operationId, stepTask->toString().c_str());
        }

        if (stepTask->isCommandTask()) {
            uint32_t commandChannelId = getTaskRunner()->getConfiguration()->getCommandChannel(stepId);
            addToCommandTrace(CAST_COMMANDTASK(stepTask), commandChannelId);
        }
    }

    void STTaskContext::onOperationPost(STNormalTask* opTask)
    {
        if (nullptr == opTask
            || nullptr == getTaskRunner()
            || nullptr == getTaskRunner()->getConfiguration()) {
            STLOG_E("Memory error.");
            return;
        }

        uint32_t operationId = opTask->getOperationId();

        std::lock_guard<std::mutex> autoSync(m_syncObj);
        addToOperationTrace(opTask, operationId);
    }

    void STTaskContext::onStepStart(STStepTask* stepTask)
    {
        if (nullptr == stepTask
            || nullptr == getTaskRunner()
            || nullptr == getTaskRunner()->getConfiguration()) {
            STLOG_E("Memory error.");
            return;
        }

        uint32_t stepId = stepTask->getStepId();
        // uint32_t operationId = stepTask->getOperationId();

        if (stepTask->isCommandTask()) {
            STCommandTask* cmdTask = CAST_COMMANDTASK(stepTask);
            if (nullptr == cmdTask) {
                return;
            }

            uint32_t commandId = stepId;

            uint32_t commandChannelId = getTaskRunner()->getConfiguration()->getCommandChannel(commandId);

            std::lock_guard<std::mutex> autoSync(m_syncObj);
            if (m_commandTraces.count(commandChannelId) > 0) {
                if (nullptr == m_commandTraces[commandChannelId].executing) {
                    m_commandTraces[commandChannelId].executing = cmdTask;
                    // STLOG_D("CommandTrace(%d) set executing from nullptr to Task(%s).", commandChannelId, cmdTask->toString().c_str());
                }
                else {
                    STLOG_E("CommandTrace(%d) must be null.", commandChannelId);
                }
            }
            else {
                STLOG_E("CommandTrace(%d) not exist!", commandChannelId);
            }
        }
    }

    void STTaskContext::onOperationStart(STNormalTask* opTask)
    {
        if (nullptr == opTask
            || nullptr == getTaskRunner()
            || nullptr == getTaskRunner()->getConfiguration()) {
            STLOG_E("Memory error.");
            return;
        }

        uint32_t operationId = opTask->getOperationId();

        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            if (nullptr == m_operationTraces[operationId].curOperationTask) {
                m_operationTraces[operationId].curOperationTask = opTask;
                // STLOG_D("OperationTrace(%d) set curOperationTask from nullptr to Task(%s).", operationId, opTask->toString().c_str());
            }
            else {
                STLOG_E("OperationTrace(%d)'s  curOperationTask must be nullptr, but now is Task(%s).", operationId, m_operationTraces[operationId].curOperationTask->toString().c_str());
            }
        }
    }

    void STTaskContext::onStepEnd(STStepTask* stepTask, uint32_t result)
    {
        (void)(result);
        onStepDelete(stepTask);
    }

    void STTaskContext::onOperationEnd(STNormalTask* opTask, uint32_t result)
    {
        (void)(result);
        onOperationDelete(opTask);
    }

    void STTaskContext::onStepDelete(STStepTask* stepTask)
    {
        if (nullptr == stepTask
            || nullptr == getTaskRunner()
            || nullptr == getTaskRunner()->getConfiguration()) {
            STLOG_E("Memory error.");
            return;
        }

        uint32_t stepId = stepTask->getStepId();
        uint32_t operationId = stepTask->getOperationId();

        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            if (m_operationTraces[operationId].curStepTask == stepTask) {
                m_operationTraces[operationId].curStepTask = nullptr;
                // STLOG_D("OperationTrace(%d) set curStepTask from Task(%s) to nullptr.", operationId, stepTask->toString().c_str());
            }
            else {
                if (nullptr == m_operationTraces[operationId].curStepTask) {
                    // STLOG_W("OperationTrace(%d)'s curStepTask nullptr.");
                }
                else {
                    // STLOG_W("OperationTrace(%d)'s curStepTask is not Task(%s) or nullptr.", operationId, stepTask->toString().c_str());
                }
            }
        }

        if (stepTask->isCommandTask()) {
            STCommandTask* cmdTask = CAST_COMMANDTASK(stepTask);
            if (nullptr == cmdTask) {
                STLOG_E("Memory error.");
                return;
            }

            uint32_t commandChannelId = getTaskRunner()->getConfiguration()->getCommandChannel(stepId);

            if (m_commandTraces.count(commandChannelId) > 0) {
                if (cmdTask == m_commandTraces[commandChannelId].executing) {
                    m_commandTraces[commandChannelId].executing = nullptr;
                    // STLOG_D("CommandTrace(%d) set executing from Task(%s) to nullptr.", commandChannelId, cmdTask->toString().c_str());
                }
                else {
                    if (nullptr == m_commandTraces[commandChannelId].executing) {
                        // STLOG_D("CommandTrace(%d)'s executing is nullptr.", commandChannelId);
                    }
                    else {
                        STLOG_W("CommandTrace(%d)'s executing is not Task(%s) or nullptr.", commandChannelId, cmdTask->toString().c_str());
                    }

                }
            }
            else {
                STLOG_E("CommandTrace(%d) not exist!", commandChannelId);
            }

            removeFromCommandTrace(cmdTask, commandChannelId);
        }
    }

    void STTaskContext::onOperationDelete(STNormalTask* opTask)
    {
        if (nullptr == opTask) {
            STLOG_E("Memory error.");
            return;
        }

        uint32_t operationId = opTask->getOperationId();

        std::lock_guard<std::mutex> autoSync(m_syncObj);
        if (m_operationTraces.count(operationId) > 0) {
            if (opTask != m_operationTraces[operationId].curOperationTask) {
                if (nullptr == m_operationTraces[operationId].curOperationTask) {
                    // STLOG_D("OperationTrace(%d)'s curOperationTask is nullptr.", operationId);
                }
                else {
                    STLOG_W("OperationTrace(%d)'s curOperationTask is not Task(%s) or nullptr.", operationId, opTask->toString().c_str());
                }
            }
            else {
                m_operationTraces[operationId].curOperationTask = nullptr;
                // STLOG_D("OperationTrace(%d) set curOperationTask from Task(%s) to nullptr.", operationId, opTask->toString().c_str());
            }
        }

        removeFromOperationTrace(opTask, operationId);
    }


    bool STTaskContext::addToCommandTrace(STCommandTask* task, uint32_t commandChannelId)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return false;
        }

        if (m_commandTraces.count(commandChannelId) > 0) {
            const COMMAND_TRACE& commandTraceConst = m_commandTraces[commandChannelId];
            COMMAND_QUEUE& commandTaskQueue = const_cast<COMMAND_QUEUE&>(commandTraceConst.commandTasks);
            commandTaskQueue.push_back(task);
            // STLOG_D("CommandTrace(%d) append CommandTask(%s).", commandChannelId, task->toString().c_str());
            return true;
        }
        STLOG_E("CommandTrace(%d) not exist!", commandChannelId);
        return false;
    }

    void STTaskContext::removeFromCommandTrace(STCommandTask* task, uint32_t commandChannelId)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return;
        }

        if (m_commandTraces.count(commandChannelId) > 0) {
            const COMMAND_TRACE& commandTraceConst = m_commandTraces[commandChannelId];
            COMMAND_QUEUE& commandTaskQueue = const_cast<COMMAND_QUEUE&>(commandTraceConst.commandTasks);

            for (COMMAND_QUEUE::iterator it = commandTaskQueue.begin(); it != commandTaskQueue.end();) {
                STCommandTask* tmp = *it;
                if (nullptr == tmp) {
                    break;
                }

                if (tmp == task) {
                    it = commandTaskQueue.erase(it);
                    // delete tmp;
                    // tmp = nullptr;
                    // STLOG_D("CommandTrace(%d) removed Task(%s).", commandChannelId, task->toString().c_str());
                }
                else {
                    ++it;
                }
            }
        }
        else {
            STLOG_E("CommandTrace(%d) not exist!", commandChannelId);
        }
    }

    bool STTaskContext::addToOperationTrace(STNormalTask* task, uint32_t operationId)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return false;
        }

        if (m_operationTraces.count(operationId) > 0) {
            const OPERATION_TRACE& operationTraceConst = m_operationTraces[operationId];
            TASK_QUEUE& operationQueue = const_cast<TASK_QUEUE&>(operationTraceConst.allOperationTasks);
            operationQueue.push_back(task);
            // STLOG_D("OpertionTrace(%d) append Task(%s).", operationId, task->toString().c_str());
            return true;
        }
        STLOG_E("OpertionTrace(%d) not exist.", operationId);
        return false;
    }

    void STTaskContext::removeFromOperationTrace(STNormalTask* task, uint32_t operationId)
    {
        if (nullptr == task) {
            STLOG_E("Memory error.");
            return;
        }

        if (m_operationTraces.count(operationId) > 0) {
            const OPERATION_TRACE& operationTraceConst = m_operationTraces[operationId];
            TASK_QUEUE& operationQueue = const_cast<TASK_QUEUE&>(operationTraceConst.allOperationTasks);
            for (TASK_QUEUE::iterator it = operationQueue.begin(); it != operationQueue.end();) {
                STNormalTask* tmp = *it;
                if (nullptr == tmp) {
                    break;
                }

                if (tmp == task) {
                    it = operationQueue.erase(it);
                    // STLOG_D("OpertionTrace(%d) removed Task(%s).", operationId, task->toString().c_str());
                }
                else {
                    ++it;
                }
            }
        }
        else {
            STLOG_E("OpertionTrace(%d) not exist.", operationId);
        }
    }

    uint32_t STTaskContext::getOperationCommandFromQueue(uint32_t operationId, const COMMAND_QUEUE& commandQueue)
    {
        COMMAND_QUEUE& commandQueueToUse = const_cast<COMMAND_QUEUE&>(commandQueue);
        for (COMMAND_QUEUE::iterator it = commandQueueToUse.begin(); it != commandQueueToUse.end(); ++it) {
            STCommandTask* tmp = *it;
            if (nullptr == tmp) {
                break;
            }

            if (tmp->getOperationId() == operationId) {
                return tmp->getStepId();
            }
        }

        return ST_TASK_INVALID_STEP;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
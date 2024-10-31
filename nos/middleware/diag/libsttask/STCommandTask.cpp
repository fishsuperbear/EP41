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
 * @file STCommandTask.cpp
 * @brief implements of STCommandTask
 */

#include "STCommandTask.h"
#include "STLogDef.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STCommandTask::STCommandTask(uint32_t commandId, STObject* parent, STObject::TaskCB callback)
        : STStepTask(commandId, ST_TASK_TYPE_COMMAND, parent, callback)
        , m_commandStatus(COMMAND_STATUS_NOTSTARTED)
    {
    }

    STCommandTask::~STCommandTask()
    {
    }

    void STCommandTask::setTaskResult(uint32_t result)
    {
        if (eNone == result) {
            STLOG_W("Command(%s) set result eNone(%d).", toString().c_str(), result);
            m_commandStatus = COMMAND_STATUS_NOTSTARTED;
            return;
        }

        if (eContinue == result) {
            STLOG_W("Command(%s) set result eContinue(%d).", toString().c_str(), result);
            m_commandStatus = COMMAND_STATUS_WAIT;
            return;
        }

        if (eNone != m_taskResult) {
            return;
        }

        if (eInterrupt != result && isIntterrupted()) {
            // STLOG_I("set Command(%s) result eInterrupt, ignore result(%d).", toString().c_str(), result);
            m_taskResult = eInterrupt;
        }
        else {
            // STLOG_I("set Command(%s) result(%d).", toString().c_str(),  result);
            m_taskResult = result;
        }
        m_commandStatus = COMMAND_STATUS_FINISHED;
    }

    bool STCommandTask::isWaitEvent() const
    {
        return COMMAND_STATUS_WAIT == m_commandStatus ? true : false;
    }

    bool STCommandTask::isExecuted() const
    {
        return COMMAND_STATUS_NOTSTARTED == m_commandStatus ? false : true;
    }

    std::string STCommandTask::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), "%s[%p, %d, %d]", getObjectName().c_str(), this, getOperationId(), getStepId());
        val.assign(buf, strlen(buf));
        return val;
    }

    std::string STCommandTask::getObjectName()
    {
        return std::string("Command");
    }

    bool STCommandTask::onStepEvent(bool isTimeout, STEvent* event)
    {
        return onCommandEvent(isTimeout, event);
    }

    uint32_t STCommandTask::doAction()
    {
        if (COMMAND_STATUS_NOTSTARTED != m_commandStatus) {
            m_commandStatus = COMMAND_STATUS_FINISHED;
            return eError;
        }

        m_commandStatus = COMMAND_STATUS_EXECUTED;

        uint32_t result = doCommand();

        if (result != eContinue) {
            m_commandStatus = COMMAND_STATUS_FINISHED;
        }
        return result;
    }

    bool STCommandTask::onCommandInterruptCheck(uint32_t interruptReason)
    {
        (void)(interruptReason);
        return true;
    }

    bool STCommandTask::waitEvent(uint32_t timeout)
    {
        if (COMMAND_STATUS_EXECUTED == m_commandStatus
            || COMMAND_STATUS_WAIT == m_commandStatus) {
            if (isIntterrupted()) {
                setTaskResult(eInterrupt);
                return false;
            }

            if (restartTimer(timeout)) {
                m_commandStatus = COMMAND_STATUS_WAIT;
                return true;
            }
            STLOG_E("Command(%s) fail to restartTimer(%d).", toString().c_str(), timeout);
        }
        return false;
    }

    bool STCommandTask::checkOnIntterupt(uint32_t interruptReason)
    {
        if (COMMAND_STATUS_WAIT == m_commandStatus) {
            if (false == onCommandInterruptCheck(interruptReason)) {
                return false;
            }
        }
        return true;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */

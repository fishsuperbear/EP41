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
 * @file STStepTask.cpp
 * @brief implements of STStepTask
 */

#include "STStepTask.h"
#include "STLogDef.h"
#include "STTaskRunner.h"
#include "STEvent.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STStepTask::STStepTask(uint32_t stepId, ST_TASK_TYPE taskType, STObject* parent, STObject::TaskCB callback)
        : STTask(taskType, ST_TASK_INVALID_OPERATION, parent, callback, false)
        , m_stepId(stepId)
        , m_scheduledEventId(0)
    {
        // Nothing here
    }

    STStepTask::~STStepTask()
    {
        unscheduleEvent(m_scheduledEventId);
    }

    bool STStepTask::onEvent(STEvent* event)
    {
        if (STEvent::isTimerEvent(m_scheduledEventId, event)) {
            m_scheduledEventId = 0;
            return onStepEvent(true, event);
        }
        return onStepEvent(false, event);
    }

    uint32_t STStepTask::getStepId() const
    {
        return m_stepId;
    }

    void STStepTask::onCallbackAction(uint32_t result)
    {
        (void)(result);
    }

    bool STStepTask::setInterrupted(uint32_t interruptReason)
    {
        if (ST_TASK_INTERRUPTREASON_INVALID == interruptReason) {
            // invalid intteruptReason
            return false;
        }
        if (isIntterrupted()) {
            // already interrupted
            return false;
        }
        onInterrupt(interruptReason);
        return true;
    }

    bool STStepTask::onStepEvent(bool isTimeout, STEvent* event)
    {
        (void)(isTimeout);
        (void)(event);
        return false;
    }

    std::string STStepTask::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), "%s[%p, %d, %d]", getObjectName().c_str(), this, getOperationId(), getStepId());
        val.assign(buf, strlen(buf));
        return val;
    }

    std::string STStepTask::getObjectName()
    {
        if (isTimerTask()) {
            return std::string("Timer");
        }
        else if (isCommandTask()) {
            return std::string("Command");
        }
        else {
            return std::string("Step");
        }
    }

    void STStepTask::onInterruptAction(uint32_t interruptReason)
    {
        if (false == checkOnIntterupt(interruptReason)) {
            return;
        }
        setTaskResult(eInterrupt);
    }

    bool STStepTask::checkOnIntterupt(uint32_t interruptReason)
    {
        (void)(interruptReason);
        return true;
    }

    bool STStepTask::restartTimer(uint32_t timeout)
    {
        if (0 == timeout) {
            return false;
        }
        unscheduleEvent(m_scheduledEventId);
        m_scheduledEventId = scheduleEvent(timeout);
        STLOG_D("restartTimer timeId: %d, timeout: %d", m_scheduledEventId, timeout);
        return 0 == m_scheduledEventId ? false : true;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
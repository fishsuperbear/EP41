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
 * @file STTimerTask.cpp
 * @brief implements of STTimerTask
 */

#include "STTimerTask.h"
#include <string.h>

namespace hozon {
namespace netaos {
namespace sttask {

    STTimerTask::STTimerTask(uint32_t timerId, STObject* parent, STObject::TaskCB callback, uint32_t timeoutInMs)
        : STStepTask(timerId, ST_TASK_TYPE_TIMER, parent, callback)
        , m_timeout(timeoutInMs)
    {
    }

    STTimerTask::~STTimerTask()
    {
    }


    bool STTimerTask::isWaitEvent() const
    {
        return eNone != getTaskResult() ? true : false;
    }

    std::string STTimerTask::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), "Timer[%p, %X, %X]", this, getOperationId(), getStepId());
        val.assign(buf, strlen(buf));
        return val;
    }

    bool STTimerTask::onStepEvent(bool isTimeout, STEvent* event)
    {
        (void)(isTimeout);
        (void)(event);
        return onTimerEvent(isTimeout, event);
    }

    uint32_t STTimerTask::doAction()
    {
        if (restartTimer(m_timeout)) {
            return eContinue;
        }
        STLOG_E("Timer(%s) fail to restartTimer(%d).", toString().c_str(), m_timeout);
        return eError;
    }

    bool STTimerTask::onTimerInterruptCheck(uint32_t interruptReason)
    {
        (void)(interruptReason);
        return true;
    }

    bool STTimerTask::checkOnIntterupt(uint32_t interruptReason)
    {
        if (!onTimerInterruptCheck(interruptReason)) {
            return false;
        }
        return true;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class TimerTaskBase implement
 */

#include "timer_task_base.h"

namespace hozon {
namespace netaos {
namespace diag {

    TimerTaskBase::TimerTaskBase(uint32_t timerId, STObject* parent, STObject::TaskCB callback, uint32_t timeout)
        : STTimerTask(timerId, parent, callback, timeout)
    {
    }

    TimerTaskBase::~TimerTaskBase()
    {
    }

    bool TimerTaskBase::onTimerEvent(bool isTimeout, STEvent* event)
    {
        return onEventAction(isTimeout, event);
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
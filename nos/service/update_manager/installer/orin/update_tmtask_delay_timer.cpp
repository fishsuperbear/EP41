/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class EthCommandTask implement
 */

#include "update_tmtask_delay_timer.h"

namespace hozon {
namespace netaos {
namespace update {

    UpdateTMTaskDelayTimer::UpdateTMTaskDelayTimer(uint32_t timerId, STObject* parent, STObject::TaskCB callback, uint32_t timeout)
        : TimerTaskBase(timerId, parent, callback, timeout)
    {
    }

    UpdateTMTaskDelayTimer::~UpdateTMTaskDelayTimer()
    {
    }

    bool UpdateTMTaskDelayTimer::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            setTaskResult(N_OK);
            return true;
        }
        return false;
    }


} // end of update
} // end of netaos
} // end of hozon
/* EOF */
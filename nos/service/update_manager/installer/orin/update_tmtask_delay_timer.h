/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateTMTaskDelayTimer Header
 */

#ifndef UPDATE_TMTASK_DELAY_TIMER_H_
#define UPDATE_TMTASK_DELAY_TIMER_H_

#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "update_manager/taskbase/timer_task_base.h"
#include "update_manager/taskbase/task_object_def.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief Class of UpdateTMTaskDelayTimer
     *
     * This class is a command task.
     */
    class UpdateTMTaskDelayTimer : public TimerTaskBase
    {
    public:
        UpdateTMTaskDelayTimer(uint32_t timerId, STObject* parent, STObject::TaskCB callback, uint32_t timeout);
        virtual ~UpdateTMTaskDelayTimer();

    protected:
        virtual bool         onEventAction(bool isTimeout, STEvent* event);

    private:
        UpdateTMTaskDelayTimer(const UpdateTMTaskDelayTimer&);
        UpdateTMTaskDelayTimer& operator=(const UpdateTMTaskDelayTimer&);
    };

} // end of update
} // end of netaos
} // end of hozon

#endif /* UPDATE_TMTASK_DELAY_TIMER_H_ */
/* EOF */
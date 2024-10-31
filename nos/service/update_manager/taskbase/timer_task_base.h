/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class TimerTaskBase Header
 */

#ifndef TIMER_TASK_BASE_H_
#define TIMER_TASK_BASE_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "diag/libsttask/STTimerTask.h"

using namespace hozon::netaos::sttask;

namespace hozon {
namespace netaos {
namespace update {
    /**
     * @brief Class of TimerTaskBase
     *
     * This class is a base task.
     */
    class TimerTaskBase : public STTimerTask
    {
    public:
        TimerTaskBase(uint32_t timerId, STObject* parent, STObject::TaskCB callback, uint32_t timeout);
        virtual ~TimerTaskBase();

    protected:
        virtual bool         onTimerEvent(bool isTimeout, STEvent* event);
        virtual bool         onEventAction(bool isTimeout, STEvent* event) = 0;

    private:
        TimerTaskBase(const TimerTaskBase&);
        TimerTaskBase& operator=(const TimerTaskBase&);
    };

} // end of diag
} // end of netaos
} // end of update
#endif  // TIMER_TASK_BASE_H_
/* EOF */

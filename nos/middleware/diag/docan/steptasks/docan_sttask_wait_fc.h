/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskWaitFC Header
 */

#ifndef DOCAN_ST_TASK_WAIT_FC_H_
#define DOCAN_ST_TASK_WAIT_FC_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "diag/docan/taskbase/command_task_base.h"
#include "diag/docan/taskbase/task_object_def.h"

namespace hozon {
namespace netaos {
namespace diag {

    /**
     * @brief class DocanSTTaskWaitFC
     *
     * Docan task class definition.
     */
    class DocanSTTaskWaitFC : public CommandTaskBase
    {
    public:
        DocanSTTaskWaitFC(STObject* pParent, STObject::TaskCB pfnCallback,
            const TaskReqInfo& reqInfo, const TaskResInfo& resInfo);
        virtual ~DocanSTTaskWaitFC();

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        DocanSTTaskWaitFC(const DocanSTTaskWaitFC&);
        DocanSTTaskWaitFC& operator=(const DocanSTTaskWaitFC&);

    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_ST_TASK_WAIT_FC_H_
/* EOF */

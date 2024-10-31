/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskEcuMonitor Header
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
     * @brief class DocanSTTaskEcuMonitor
     *
     * Docan task class definition.
     */
    class DocanSTTaskEcuMonitor : public CommandTaskBase
    {
    public:
        DocanSTTaskEcuMonitor(STObject* pParent, STObject::TaskCB pfnCallback,
            const TaskReqInfo& reqInfo, const TaskResInfo& resInfo);
        virtual ~DocanSTTaskEcuMonitor();

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        DocanSTTaskEcuMonitor(const DocanSTTaskEcuMonitor&);
        DocanSTTaskEcuMonitor& operator=(const DocanSTTaskEcuMonitor&);

    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_ST_TASK_WAIT_FC_H_
/* EOF */

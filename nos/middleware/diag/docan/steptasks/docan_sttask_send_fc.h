/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskSendFC Header
 */

#ifndef DOCAN_ST_TASK_SEND_FC_H_
#define DOCAN_ST_TASK_SEND_FC_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <vector>
#include "diag/docan/taskbase/command_task_base.h"
#include "diag/docan/taskbase/task_object_def.h"

namespace hozon {
namespace netaos {
namespace diag {

    /**
     * @brief class DocanSTTaskSendFC
     *
     * Docan task class definition.
     */
    class DocanSTTaskSendFC : public CommandTaskBase
    {
    public:
        DocanSTTaskSendFC(STObject* pParent, STObject::TaskCB pfnCallback,
            const TaskReqInfo& reqInfo, const TaskResInfo& resInfo);
        virtual ~DocanSTTaskSendFC();

    protected:
        virtual uint32_t     doCommand();
        virtual bool         onEventAction(bool isTimeout, STEvent* event);

    private:
        DocanSTTaskSendFC(const DocanSTTaskSendFC&);
        DocanSTTaskSendFC& operator=(const DocanSTTaskSendFC&);

    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_ST_TASK_SEND_FC_H_
/* EOF */

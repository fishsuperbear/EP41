/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskSendSF Header
 */

#ifndef DOCAN_ST_TASK_SEND_SF_H_
#define DOCAN_ST_TASK_SEND_SF_H_
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
     * @brief class DocanSTTaskSendSF
     *
     * Docan task class definition.
     */
    class DocanSTTaskSendSF : public CommandTaskBase
    {
    public:
        DocanSTTaskSendSF(STObject* pParent, STObject::TaskCB pfnCallback,
            const TaskReqInfo& reqInfo, const TaskResInfo& resInfo);
        virtual ~DocanSTTaskSendSF();

    protected:
        virtual uint32_t     doCommand();
        virtual bool         onEventAction(bool isTimeout, STEvent* event);

    private:
        DocanSTTaskSendSF(const DocanSTTaskSendSF&);
        DocanSTTaskSendSF& operator=(const DocanSTTaskSendSF&);
    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_ST_TASK_SEND_SF_H_
/* EOF */

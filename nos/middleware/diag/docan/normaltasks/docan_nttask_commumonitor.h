/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanNTTaskSendCommand Header
 */

#ifndef DOCAN_NT_TASK_COMMU_MONITOR_H_
#define DOCAN_NT_TASK_COMMU_MONITOR_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <vector>
#include "diag/docan/taskbase/normal_task_base.h"
#include "diag/docan/taskbase/task_object_def.h"
#include "diag/docan/common/docan_internal_def.h"

namespace hozon {
namespace netaos {
namespace diag {


    /**
     * @brief class DocanNTTaskCommuMonitor
     *
     * Docan task class definition.
     */
    class DocanNTTaskCommuMonitor : public NormalTaskBase
    {
    public:
        DocanNTTaskCommuMonitor(NormalTaskBase* pParent, STObject::TaskCB pfnCallback, const TaskReqInfo& info, bool isTopTask = false);
        virtual ~DocanNTTaskCommuMonitor();

    protected:
        virtual uint32_t        doAction();
        virtual void            onCallbackAction(uint32_t result);

        uint32_t                startToCommuMonitor();
        void                    onCommuMonitorResult(STTask* task, uint32_t result);

    private:
        DocanNTTaskCommuMonitor(const DocanNTTaskCommuMonitor&);
        DocanNTTaskCommuMonitor& operator=(const DocanNTTaskCommuMonitor&);

        TaskReqInfo         m_reqInfo;

    };

} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_NT_TASK_COMMU_MONITOR_H_
/* EOF */
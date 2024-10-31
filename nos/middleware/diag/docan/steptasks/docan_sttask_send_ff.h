/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskSendFF Header
 */

#ifndef DOCAN_ST_TASK_SEND_FF_H_
#define DOCAN_ST_TASK_SEND_FF_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "diag/docan/taskbase/command_task_base.h"
#include "diag/docan/taskbase/task_object_def.h"

namespace hozon {
namespace netaos {
namespace diag {

    /**
     * @brief class DocanSTTaskSendFF
     *
     * Docan task class definition.
     */
    class DocanSTTaskSendFF : public CommandTaskBase
    {
    public:
        DocanSTTaskSendFF(STObject* pParent, STObject::TaskCB pfnCallback,
            const TaskReqInfo& reqInfo, const TaskResInfo& resInfo);
        virtual ~DocanSTTaskSendFF();

        uint8_t             getFcStatus() const;
        uint8_t             getFcBs() const;
        uint8_t             getFcSTmin() const;

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        DocanSTTaskSendFF(const DocanSTTaskSendFF&);
        DocanSTTaskSendFF& operator=(const DocanSTTaskSendFF&);

    private:
        uint8_t             m_fcStatus;
        uint8_t             m_fcBs;
        uint8_t             m_fcSTmin;

    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_ST_TASK_SEND_FF_H_
/* EOF */

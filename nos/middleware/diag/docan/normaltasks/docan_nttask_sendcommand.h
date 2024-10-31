/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanNTTaskSendCommand Header
 */

#ifndef DOCAN_NT_TASK_SENDCOMMAND_H_
#define DOCAN_NT_TASK_SENDCOMMAND_H_
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
     * @brief class DocanNTTaskSendCommand
     *
     * Docan task class definition.
     */
    class DocanNTTaskSendCommand : public NormalTaskBase
    {
    public:
        /**
         * @brief FlowStatus
         */
        typedef enum {
            CTS   = 0x0,
            WAIT  = 0x1,
            OVFLW = 0x2
        } FS_t;

        DocanNTTaskSendCommand(NormalTaskBase* pParent, STObject::TaskCB pfnCallback, const TaskReqInfo& reqInfo,
            bool isTopTask = false);
        virtual ~DocanNTTaskSendCommand();

        virtual TaskReqInfo&        getReqInfo();
        virtual TaskResInfo&        getResInfo();

    protected:
        virtual uint32_t        doAction();
        virtual void            onCallbackAction(uint32_t taskResult);

        uint32_t                startToSendSF();
        void                    onSendSFResult(STTask *task, uint32_t result);

        uint32_t                startToSendFF();
        void                    onSendFFResult(STTask *task, uint32_t result);

        uint32_t                startToWaitFC();
        void                    onWaitFCResult(STTask *task, uint32_t result);

        uint32_t                startToSendCF();
        void                    onSendCFResult(STTask *task, uint32_t result);

        uint32_t                startToWaitPending();
        void                    onWaitPendingResult(STTask *task, uint32_t result);

        uint32_t                startToSendFC();
        void                    onSendFCResult(STTask *task, uint32_t result);

    private:
        DocanNTTaskSendCommand(const DocanNTTaskSendCommand&);
        DocanNTTaskSendCommand& operator=(const DocanNTTaskSendCommand&);

    private:
        TaskReqInfo             m_reqInfo;
        TaskResInfo             m_resInfo;

        N_EcuInfo_t             m_ecuInfo;
        uint8_t                 m_waitFcRetry;
    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_NT_TASK_SENDCOMMAND_H_
/* EOF */

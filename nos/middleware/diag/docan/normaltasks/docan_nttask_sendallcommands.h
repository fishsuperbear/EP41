/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanNTTaskSendCommand Header
 */

#ifndef DOCAN_NT_TASK_SEND_ALL_COMMANDS_H_
#define DOCAN_NT_TASK_SEND_ALL_COMMANDS_H_
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
     * @brief class DocanNTTaskSendAllCommands
     *
     * Docan task class definition.
     */
    class DocanNTTaskSendAllCommands : public NormalTaskBase
    {
    public:
        DocanNTTaskSendAllCommands(uint32_t taskType, NormalTaskBase* pParent, STObject::TaskCB pfnCallback, bool isTopTask = false, bool isBlockNext = true);
        virtual ~DocanNTTaskSendAllCommands();

    protected:
        virtual uint32_t        doAction();
        virtual void            onCallbackAction(uint32_t result);

        uint32_t                startToSendCommand();
        void                    onSendCommandResult(STTask* task, uint32_t result);

        void                    addCommand(const TaskReqInfo& commandInfo);
        virtual uint32_t        onCommandResponse(const uint32_t commandIndex, STTask* task, uint32_t result);

    private:
        DocanNTTaskSendAllCommands(const DocanNTTaskSendAllCommands&);
        DocanNTTaskSendAllCommands& operator=(const DocanNTTaskSendAllCommands&);

        std::vector<TaskReqInfo>    m_allCommands;
        uint32_t                    m_commandIndex;
        bool                        m_isBlockNext;

    };

} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_NT_TASK_SEND_ALL_COMMANDS_H_
/* EOF */
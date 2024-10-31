/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanNTTaskSendCommand implement
 */

#include "docan_nttask_sendallcommands.h"
#include "docan_nttask_sendcommand.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanNTTaskSendAllCommands::DocanNTTaskSendAllCommands(uint32_t taskType, NormalTaskBase* pParent, STObject::TaskCB pfnCallback, bool isTopTask,
        bool isBlockNext)
        : NormalTaskBase(taskType, pParent, pfnCallback, isTopTask)
        , m_commandIndex(0)
        , m_isBlockNext(isBlockNext)
    {
    }

    DocanNTTaskSendAllCommands::~DocanNTTaskSendAllCommands()
    {
    }

    uint32_t DocanNTTaskSendAllCommands::doAction()
    {
        return startToSendCommand();
    }

    void DocanNTTaskSendAllCommands::onCallbackAction(uint32_t result)
    {
    }

    uint32_t DocanNTTaskSendAllCommands::startToSendCommand()
    {
        if (m_commandIndex < m_allCommands.size()) {

            TaskReqInfo cmdInfo = m_allCommands[m_commandIndex];

            DocanNTTaskSendCommand* task = new DocanNTTaskSendCommand(this, CAST_TASK_CB(&DocanNTTaskSendAllCommands::onSendCommandResult), cmdInfo);

            return post(task);
        }

        return eOK;
    }

    void DocanNTTaskSendAllCommands::onSendCommandResult(STTask* task, uint32_t result)
    {
        m_commandIndex++;

        if (N_OK != result && m_isBlockNext) {
            onCallbackResult(result);
            return;
        }

        DocanNTTaskSendCommand* targetTask = static_cast<DocanNTTaskSendCommand*>(task);
        if (nullptr== targetTask) {
            onCallbackResult(N_ERROR);
            return;
        }

        result = onCommandResponse((m_commandIndex - 1), task, result);
        if (N_OK != result && m_isBlockNext) {
            onCallbackResult(result);
            return;
        }

        uint32_t res = startToSendCommand();
        if (eContinue != res) {
            onCallbackResult(res);
        }
    }

    void DocanNTTaskSendAllCommands::addCommand(const TaskReqInfo& commandInfo)
    {
        m_allCommands.push_back(commandInfo);
    }

    uint32_t DocanNTTaskSendAllCommands::onCommandResponse(const uint32_t commandIndex, STTask* task, uint32_t result)
    {
        return eOK;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */

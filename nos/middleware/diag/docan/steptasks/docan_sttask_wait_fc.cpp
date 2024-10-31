/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskWaitFC implement
 */

#include "docan_sttask_wait_fc.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanSTTaskWaitFC::DocanSTTaskWaitFC(STObject* pParent, STObject::TaskCB pfnCallback,
        const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : CommandTaskBase(DOCAN_COMMAND_WAIT_FC + reqInfo.reqEcu, pParent, pfnCallback, reqInfo, resInfo)
    {
    }

    DocanSTTaskWaitFC::~DocanSTTaskWaitFC()
    {
    }

    uint32_t DocanSTTaskWaitFC::doCommand()
    {
        if (waitEvent(DOCAN_TIMER_Bs)) {
            return eContinue;
        }

        return N_TIMEOUT_Bs;
    }

    bool DocanSTTaskWaitFC::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            setTaskResult(N_TIMEOUT_Bs);
            return true;
        }

        DocanTaskEvent* taskevent = static_cast<DocanTaskEvent*>(event);
        if (nullptr == taskevent) {
            return false;
        }

        if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, m_resInfo.resCanid, N_PCItype_FC, event)) {
            m_resInfo.resFs = m_resInfo.resContent[0] & 0x0F;
            m_resInfo.resBs = m_resInfo.resContent[1];
            m_resInfo.resSTmin = m_resInfo.resContent[2];
            if (m_resInfo.resFs > 2 || m_resInfo.resSTmin > 0x7F) {
                DOCAN_LOG_D("invalid FC frame.");
                setTaskResult(N_TIMEOUT_Bs);
            }
            else {
                setTaskResult(N_OK);
            }
            return true;
        }

        return false;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */

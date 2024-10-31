/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSTTaskEcuMonitor implement
 */

#include "docan_sttask_ecu_monitor.h"
#include "diag/docan/taskbase/docan_task_event.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanSTTaskEcuMonitor::DocanSTTaskEcuMonitor(STObject* pParent, STObject::TaskCB pfnCallback,
        const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : CommandTaskBase(DOCAN_COMMAND_ECU_MONITOR + reqInfo.reqEcu, pParent, pfnCallback, reqInfo, resInfo)
    {
    }

    DocanSTTaskEcuMonitor::~DocanSTTaskEcuMonitor()
    {
    }

    uint32_t DocanSTTaskEcuMonitor::doCommand()
    {
        if (waitEvent(DOCAN_TIMER_CommuMonitor)) {
            return eContinue;
        }

        return N_ERROR;
    }

    bool DocanSTTaskEcuMonitor::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            setTaskResult(N_OK);
            return true;
        }

        DocanTaskEvent* taskevent = static_cast<DocanTaskEvent*>(event);
        if (nullptr == taskevent) {
            return false;
        }

        if (DocanTaskEvent::checkEvent(DOCAN_EVENT_ECU, m_reqInfo.reqEcu, event)) {
            setTaskResult(N_USER_CANCEL);
        }

        return false;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */

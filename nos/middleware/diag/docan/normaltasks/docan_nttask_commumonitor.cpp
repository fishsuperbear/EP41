/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanNTTaskSendCommand implement
 */

#include "docan_nttask_commumonitor.h"
#include "diag/docan/steptasks/docan_sttask_ecu_monitor.h"
#include "diag/docan/manager/docan_sys_interface.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanNTTaskCommuMonitor::DocanNTTaskCommuMonitor(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,  const TaskReqInfo& info, bool isTopTask)
        : NormalTaskBase(DOCAN_NTTASK_COMMU_MONITOR + info.reqEcu, pParent, pfnCallback, isTopTask)
        , m_reqInfo(info)
    {
    }

    DocanNTTaskCommuMonitor::~DocanNTTaskCommuMonitor()
    {
    }

    uint32_t DocanNTTaskCommuMonitor::doAction()
    {
        return startToCommuMonitor();
    }

    void DocanNTTaskCommuMonitor::onCallbackAction(uint32_t result)
    {
        if (N_OK == result) {
            DOCAN_LOG_D("ecu: %d, reqCanid: %X, communication finished, monitor %dms to close channel.", m_reqInfo.reqEcu, m_reqInfo.reqCanid, DOCAN_TIMER_CommuMonitor);
            DocanSysInterface::instance()->StopChannel(m_reqInfo.reqEcu);
        }
    }

    uint32_t DocanNTTaskCommuMonitor::startToCommuMonitor()
    {
        TaskResInfo resInfo;
        DocanSTTaskEcuMonitor* task = new DocanSTTaskEcuMonitor(this,
                                                        CAST_TASK_CB(&DocanNTTaskCommuMonitor::onCommuMonitorResult),
                                                        m_reqInfo, resInfo);
        return post(task);
    }

    void DocanNTTaskCommuMonitor::onCommuMonitorResult(STTask* task, uint32_t result)
    {
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */

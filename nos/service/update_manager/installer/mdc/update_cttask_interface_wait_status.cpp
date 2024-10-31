/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceWaitStatus implement
 */

#include "update_cttask_interface_wait_status.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/common/data_def.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

    UpdateCTTaskInterfaceWaitStatus::UpdateCTTaskInterfaceWaitStatus(STObject* pParent, STObject::TaskCB pfnCallback)
        : CommandTaskBase(OTA_CTTASK_INTERFACE_WAIT, pParent, pfnCallback)
        , m_updateStatus("")
    {

    }

    UpdateCTTaskInterfaceWaitStatus::~UpdateCTTaskInterfaceWaitStatus()
    {
    }

    const std::string& UpdateCTTaskInterfaceWaitStatus::GetUpdateStatus()
    {
        return m_updateStatus;
    }

    uint32_t UpdateCTTaskInterfaceWaitStatus::doCommand()
    {
        if (waitEvent(OTA_TIMER_P2START_CLIENT)) {
            return eContinue;
        }
        return N_ERROR;
    }

    bool UpdateCTTaskInterfaceWaitStatus::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            OTAAgent::Instance()->Query(m_updateStatus);
            if ("kActivated" == m_updateStatus || "kIdle" == m_updateStatus || "kReady" == m_updateStatus) {
                // update finish need do activate
                setTaskResult(N_OK);
            }
            else {
                // for current is updating just need wait update completed
                setTaskResult(N_WAIT);
            }
            return true;
        }

        return false;
    }

} // end of update
} // end of netaos
} // end of hozon
/* EOF */

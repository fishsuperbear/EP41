/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceActivate implement
 */

#include "update_cttask_interface_activate.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/common/data_def.h"
#include "update_manager/log/update_manager_logger.h"


namespace hozon {
namespace netaos {
namespace update {

    UpdateCTTaskInterfaceActivate::UpdateCTTaskInterfaceActivate(STObject* pParent, STObject::TaskCB pfnCallback)
        : CommandTaskBase(OTA_CTTASK_INTERFACE_ACTIVATE, pParent, pfnCallback)
    {

    }

    UpdateCTTaskInterfaceActivate::~UpdateCTTaskInterfaceActivate()
    {
    }

    uint32_t UpdateCTTaskInterfaceActivate::doCommand()
    {
        // only kIdle could do update
        if (!OTAAgent::Instance()->Activate()) {
            return N_ERROR;
        }

        if (waitEvent(OTA_TIMER_P2START_CLIENT)) {
            return eContinue;
        }
        return N_ERROR;
    }

    bool UpdateCTTaskInterfaceActivate::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            std::string updateStatus;
            OTAAgent::Instance()->Query(updateStatus);
            if ("kActivating" == updateStatus) {
                UPDATE_LOG_D("update activating, will power off restart to continue update.");
                setTaskResult(N_WAIT);
            }
            else if ("kActivated" == updateStatus){
                // activate action finish need do finish
                setTaskResult(N_OK);
            }
            else {
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

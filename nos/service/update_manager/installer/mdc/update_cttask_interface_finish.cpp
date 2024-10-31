/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceFinish implement
 */

#include "update_cttask_interface_finish.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/common/data_def.h"
#include "update_manager/log/update_manager_logger.h"


namespace hozon {
namespace netaos {
namespace update {

    UpdateCTTaskInterfaceFinish::UpdateCTTaskInterfaceFinish(STObject* pParent, STObject::TaskCB pfnCallback)
        : CommandTaskBase(OTA_CTTASK_INTERFACE_FINISH, pParent, pfnCallback)
    {

    }

    UpdateCTTaskInterfaceFinish::~UpdateCTTaskInterfaceFinish()
    {
    }

    uint32_t UpdateCTTaskInterfaceFinish::doCommand()
    {
        if (!OTAAgent::Instance()->Finish()) {
            return N_ERROR;
        }

        if (waitEvent(OTA_TIMER_P2START_CLIENT)) {
            return eContinue;
        }
        return N_ERROR;
    }

    bool UpdateCTTaskInterfaceFinish::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            std::string updateStatus;
            OTAAgent::Instance()->Query(updateStatus);
            if ("kIdle" == updateStatus || "kServiceNotAvailable" == updateStatus || "" == updateStatus) {
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

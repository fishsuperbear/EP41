/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceUpdata implement
 */

#include "update_cttask_interface_update.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/common/data_def.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

    UpdateCTTaskInterfaceUpdata::UpdateCTTaskInterfaceUpdata(STObject* pParent, STObject::TaskCB pfnCallback,
                                   const std::string& package)
        : CommandTaskBase(OTA_CTTASK_INTERFACE_UPDATE, pParent, pfnCallback)
        , m_updateStatus("")
        , m_package(package)
    {

    }

    UpdateCTTaskInterfaceUpdata::~UpdateCTTaskInterfaceUpdata()
    {
    }

    uint32_t UpdateCTTaskInterfaceUpdata::doCommand()
    {
        // only kIdle could do update
        if (!OTAAgent::Instance()->Update(m_package)) {
            return N_ERROR;
        }

        if (waitEvent(OTA_TIMER_P2START_CLIENT)) {
            return eContinue;
        }
        return N_ERROR;
    }

    bool UpdateCTTaskInterfaceUpdata::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            OTAAgent::Instance()->Query(m_updateStatus);
            if ("kReady" == m_updateStatus) {
                // update action finish need do activate
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

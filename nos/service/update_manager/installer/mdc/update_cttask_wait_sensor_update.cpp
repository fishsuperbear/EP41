/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskWaitSensorUpdate implement
 */

#include "update_cttask_wait_sensor_update.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/common/data_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/record/ota_record.h"

namespace hozon {
namespace netaos {
namespace update {

    UpdateCTTaskWaitSensorUpdate::UpdateCTTaskWaitSensorUpdate(STObject* pParent, STObject::TaskCB pfnCallback)
        : CommandTaskBase(OTA_CTTASK_INTERFACE_WAIT, pParent, pfnCallback)
        , m_sensorUpdateProgress(0)
    {

    }

    UpdateCTTaskWaitSensorUpdate::~UpdateCTTaskWaitSensorUpdate()
    {
    }

    uint8_t UpdateCTTaskWaitSensorUpdate::GetSensorUpdateProgress()
    {
        return m_sensorUpdateProgress;
    }

    uint32_t UpdateCTTaskWaitSensorUpdate::doCommand()
    {
        if (waitEvent(OTA_TIMER_P2START_CLIENT)) {
            return eContinue;
        }
        return N_ERROR;
    }

    bool UpdateCTTaskWaitSensorUpdate::onEventAction(bool isTimeout, STEvent* event)
    {
        if (isTimeout) {
            m_sensorUpdateProgress = OTARecoder::Instance().GetSensorTotalProgress();
            if (OTARecoder::Instance().IsSensorUpdateCompleted()) {
                // all Sensor update completed, but some failed.
                if (100 == m_sensorUpdateProgress) {
                    UPDATE_LOG_D("all sensor update completed.");
                    setTaskResult(N_OK);
                }
                else if (0 == m_sensorUpdateProgress) {
                    setTaskResult(N_OK);
                }
                else {
                    UPDATE_LOG_D("all sensor update completed, but some failed, ecu progress: %d%%", m_sensorUpdateProgress);
                    setTaskResult(N_ERROR);
                }
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

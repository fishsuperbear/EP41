/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskWaitSensorUpdate Header
 */

#ifndef UPDATE_CTTASK_WAIT_SENSOR_UPDATE_H_
#define UPDATE_CTTASK_WAIT_SENSOR_UPDATE_H_

#include "update_manager/taskbase/command_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class UpdateCTTaskWaitSensorUpdate
     *
     * Docan task class definition.
     */
    class UpdateCTTaskWaitSensorUpdate : public CommandTaskBase {
    public:
        UpdateCTTaskWaitSensorUpdate(STObject* pParent, STObject::TaskCB pfnCallback);
        virtual ~UpdateCTTaskWaitSensorUpdate();

        uint8_t GetSensorUpdateProgress();

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        UpdateCTTaskWaitSensorUpdate(const UpdateCTTaskWaitSensorUpdate&);
        UpdateCTTaskWaitSensorUpdate& operator=(const UpdateCTTaskWaitSensorUpdate&);

        uint8_t         m_sensorUpdateProgress;
    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // UPDATE_CTTASK_WAIT_Sensor_UPDATE_H_
/* EOF */

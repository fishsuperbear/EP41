/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: sensor update install task
 */
#ifndef UPDATE_NORMAL_TASK_SENSOR_INSTALLER_H_
#define UPDATE_NORMAL_TASK_SENSOR_INSTALLER_H_

#include "update_manager/taskbase/normal_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/config/config_manager.h"

namespace hozon {
namespace netaos {
namespace update {

class UpdateNTTaskSensorInstaller : public NormalTaskBase {
public:
    UpdateNTTaskSensorInstaller(NormalTaskBase* pParent, STObject::TaskCB pfnCallback, const Sensor_t& sensor, bool isTopTask = false);
    ~UpdateNTTaskSensorInstaller();

    virtual uint32_t        doAction();
    virtual void            onCallbackAction(uint32_t result);

    uint32_t                StartToUpdateProcess();

    uint32_t                StartToSendCommand();
    void                    OnSendCommandResult(STTask *task, uint32_t result);

    uint32_t                StartToSecurityAccess();
    void                    OnSecurityAccessResult(STTask *task, uint32_t result);

    uint32_t                StartToTransFile();
    void                    OnTransFileResult(STTask *task, uint32_t result);

    uint32_t                StartToFailRetry();
    void                    OnFailRetryResult(STTask *task, uint32_t result);

private:
    UpdateNTTaskSensorInstaller(const UpdateNTTaskSensorInstaller &);
    UpdateNTTaskSensorInstaller & operator = (const UpdateNTTaskSensorInstaller &);

    uint32_t                GetTickCount();

    uint32_t                index_;
    Sensor_t                sensor_;
    uint32_t                performance_;
    uint8_t                 progress_;
    uint8_t                 retry_;

};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_NORMAL_TASK_SENSOR_INSTALLER_H_

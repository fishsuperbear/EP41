/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: transfer file normal task
 */
#ifndef UPDATE_NTTASK_SEND_COMMANDS_H_
#define UPDATE_NTTASK_SEND_COMMANDS_H_


#include <string>

#include "update_manager/taskbase/normal_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/config/config_manager.h"

namespace hozon {
namespace netaos {
namespace update {

class UpdateNTTaskSendCommands : public NormalTaskBase {
public:
    UpdateNTTaskSendCommands(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
        const SensorInfo_t& sensorInfo, const UpdateCase_t& process, bool isTopTask = false);
    ~UpdateNTTaskSendCommands();

    TaskReqInfo&            GetReqInfo();
    TaskResInfo&            GetResInfo();

    virtual uint32_t        doAction();
    virtual void            onCallbackAction(uint32_t result);

    uint32_t                StartToSendUdsCommand();
    void                    OnSendUdsCommandResult(STTask *task, uint32_t result);

    uint32_t                StartToWaitDelay();
    void                    OnWaitDelayResult(STTask *task, uint32_t result);

private:
    UpdateNTTaskSendCommands(const UpdateNTTaskSendCommands &);
    UpdateNTTaskSendCommands & operator = (const UpdateNTTaskSendCommands &);

    /// for transfer file
    SensorInfo_t            m_sensorInfo;
    UpdateCase_t            m_process;
    TaskReqInfo             m_reqInfo;
    TaskResInfo             m_resInfo;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_NTTASK_SEND_COMMANDS_H_

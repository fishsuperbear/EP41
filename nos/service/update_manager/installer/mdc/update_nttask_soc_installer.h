/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: SoC installer normal task
 */
#ifndef UPDATE_NTTASK_SOC_INSTALLER_H_
#define UPDATE_NTTASK_SOC_INSTALLER_H_


#include "update_manager/taskbase/normal_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/config/config_manager.h"

namespace hozon {
namespace netaos {
namespace update {

class UpdateNTTaskSocInstaller : public NormalTaskBase {
public:
    UpdateNTTaskSocInstaller(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
        const SoC_t& socInfo, bool isTopTask = false);
    ~UpdateNTTaskSocInstaller();

    virtual uint32_t        doAction();
    virtual void            onCallbackAction(uint32_t result);

    uint32_t                StartToUpdateProcess();

    uint32_t                StartToWaitEcusUpdateCompleted();
    void                    OnWaitEcusUpdateCompletedResult(STTask *task, uint32_t result);

    uint32_t                StartToInterfaceUpdate();
    void                    OnInterfaceUpdateResult(STTask *task, uint32_t result);

    uint32_t                StartToInterfaceActivate();
    void                    OnInterfaceActivateResult(STTask *task, uint32_t result);

    uint32_t                StartToInterfaceFinish();
    void                    OnInterfaceFinishResult(STTask *task, uint32_t result);

    uint32_t                StartToWaitActionStatus();
    void                    OnWaitActionStatusResult(STTask *task, uint32_t result);

    uint32_t                StartToFailRetry();
    void                    OnFailRetryResult(STTask *task, uint32_t result);

private:
    UpdateNTTaskSocInstaller(const UpdateNTTaskSocInstaller &);
    UpdateNTTaskSocInstaller & operator = (const UpdateNTTaskSocInstaller &);

    uint32_t                GetTickCount();

    /// for interface installer
    SoC_t                   m_soc;

    std::string             m_updateStatus;
    uint32_t                m_retryTimes;
    uint32_t                m_updateResult;
    uint32_t                m_performance;
    uint8_t                 m_progress;
    uint8_t                 m_retry;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_NTTASK_SOC_INSTALLER_H_

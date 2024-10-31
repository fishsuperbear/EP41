/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: Security Access normal task
 */
#ifndef UPDATE_NTTASK_SECURITY_ACCESS_H_
#define UPDATE_NTTASK_SECURITY_ACCESS_H_

#include "update_manager/taskbase/normal_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/config/config_manager.h"

namespace hozon {
namespace netaos {
namespace update {

class UpdateNTTaskSecurityAccess : public NormalTaskBase {
public:
    UpdateNTTaskSecurityAccess(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
        const SensorInfo_t& header, const UpdateCase_t& process, bool isTopTask = false);
    ~UpdateNTTaskSecurityAccess();

    virtual uint32_t        doAction();
    virtual void            onCallbackAction(uint32_t result);

    uint32_t                StartToGetSeed();
    void                    OnGetSeedResult(STTask *task, uint32_t result);

    uint32_t                StartToSendKey();
    void                    OnSendKeyResult(STTask *task, uint32_t result);

private:
    UpdateNTTaskSecurityAccess(const UpdateNTTaskSecurityAccess &);
    UpdateNTTaskSecurityAccess & operator = (const UpdateNTTaskSecurityAccess &);

    int32_t     GetKeyLevel1(uint32_t& key, uint32_t seed, uint32_t APP_MASK);
    int32_t     GetKeyLevelFbl(uint32_t& key, uint32_t seed, uint32_t BOOT_MASK);

    SensorInfo_t            m_sensorInfo;
    UpdateCase_t            m_process;
    uint32_t                m_seed;

};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_NTTASK_SECURITY_ACCESS_H_

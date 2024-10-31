/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceWaitStatus Header
 */

#ifndef UPDATE_CTTASK_INTERFACE_WAIT_STATUS_H_
#define UPDATE_CTTASK_INTERFACE_WAIT_STATUS_H_

#include "update_manager/taskbase/command_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class UpdateCTTaskInterfaceWaitStatus
     *
     * Docan task class definition.
     */
    class UpdateCTTaskInterfaceWaitStatus : public CommandTaskBase {
    public:
        UpdateCTTaskInterfaceWaitStatus(STObject* pParent, STObject::TaskCB pfnCallback);
        virtual ~UpdateCTTaskInterfaceWaitStatus();

        const std::string& GetUpdateStatus();

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        UpdateCTTaskInterfaceWaitStatus(const UpdateCTTaskInterfaceWaitStatus&);
        UpdateCTTaskInterfaceWaitStatus& operator=(const UpdateCTTaskInterfaceWaitStatus&);

        std::string         m_updateStatus;
    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // UPDATE_CTTASK_INTERFACE_WAIT_STATUS_H_
/* EOF */

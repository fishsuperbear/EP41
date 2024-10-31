/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceActivate Header
 */

#ifndef UPDATE_CTTASK_INTERFACE_ACTIVATE_H_
#define UPDATE_CTTASK_INTERFACE_ACTIVATE_H_

#include "update_manager/taskbase/command_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class UpdateCTTaskInterfaceActivate
     *
     * Docan task class definition.
     */
    class UpdateCTTaskInterfaceActivate : public CommandTaskBase {
    public:
        UpdateCTTaskInterfaceActivate(STObject* pParent, STObject::TaskCB pfnCallback);
        virtual ~UpdateCTTaskInterfaceActivate();

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        UpdateCTTaskInterfaceActivate(const UpdateCTTaskInterfaceActivate&);
        UpdateCTTaskInterfaceActivate& operator=(const UpdateCTTaskInterfaceActivate&);

    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // UPDATE_CTTASK_INTERFACE_ACTIVATE_H_
/* EOF */

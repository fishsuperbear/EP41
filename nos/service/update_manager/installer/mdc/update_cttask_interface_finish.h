/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceUpdate Header
 */

#ifndef UPDATE_CTTASK_INTERFACE_FINISH_H_
#define UPDATE_CTTASK_INTERFACE_FINISH_H_


#include "update_manager/taskbase/command_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class UpdateCTTaskInterfaceFinish
     *
     * Docan task class definition.
     */
    class UpdateCTTaskInterfaceFinish : public CommandTaskBase {
    public:
        UpdateCTTaskInterfaceFinish(STObject* pParent, STObject::TaskCB pfnCallback);
        virtual ~UpdateCTTaskInterfaceFinish();

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        UpdateCTTaskInterfaceFinish(const UpdateCTTaskInterfaceFinish&);
        UpdateCTTaskInterfaceFinish& operator=(const UpdateCTTaskInterfaceFinish&);

        std::string         m_updateStatus;
    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // #ifndef UPDATE_CTTASK_INTERFACE_FINISH_H_

/* EOF */

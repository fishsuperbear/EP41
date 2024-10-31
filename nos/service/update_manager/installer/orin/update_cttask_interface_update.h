/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceUpdate Header
 */

#ifndef UPDATE_CTTASK_INTERFACE_UPDATE_H_
#define UPDATE_CTTASK_INTERFACE_UPDATE_H_

#include "update_manager/taskbase/command_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class UpdateCTTaskInterfaceUpdata
     *
     * Docan task class definition.
     */
    class UpdateCTTaskInterfaceUpdata : public CommandTaskBase {
    public:
        UpdateCTTaskInterfaceUpdata(STObject* pParent, STObject::TaskCB pfnCallback,
                       const std::string& package);
        virtual ~UpdateCTTaskInterfaceUpdata();

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        UpdateCTTaskInterfaceUpdata(const UpdateCTTaskInterfaceUpdata&);
        UpdateCTTaskInterfaceUpdata& operator=(const UpdateCTTaskInterfaceUpdata&);

        std::string         m_updateStatus;
        std::string         m_package;
    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // UPDATE_CTTASK_INTERFACE_UPDATE_H_
/* EOF */

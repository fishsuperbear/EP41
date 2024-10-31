/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskCommand Header
 */

#ifndef UPDATE_CTTASK_COMMAND_H_
#define UPDATE_CTTASK_COMMAND_H_

#include "update_manager/taskbase/command_task_base.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class UpdateCTTaskCommand
     *
     * Docan task class definition.
     */
    class UpdateCTTaskCommand : public CommandTaskBase {
    public:
        UpdateCTTaskCommand(STObject* pParent, STObject::TaskCB pfnCallback,
                       const TaskReqInfo& reqInfo, const TaskResInfo& resInfo);
        virtual ~UpdateCTTaskCommand();

        virtual TaskReqInfo&        GetReqInfo();
        virtual TaskResInfo&        GetResInfo();

    protected:
        virtual uint32_t    doCommand();
        virtual bool        onEventAction(bool isTimeout, STEvent* event);

    private:
        UpdateCTTaskCommand(const UpdateCTTaskCommand&);
        UpdateCTTaskCommand& operator=(const UpdateCTTaskCommand&);

        bool                m_suppressPositiveResponse;
        bool                m_functionAddr;
        TaskReqInfo         m_reqInfo;
        TaskResInfo         m_resInfo;
    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // UPDATE_CTTASK_COMMAND_H_
/* EOF */

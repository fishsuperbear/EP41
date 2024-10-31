/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase Header
 */

#ifndef COMMAND_TASK_BASE_H_
#define COMMAND_TASK_BASE_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif


#include "diag/libsttask/STCommandTask.h"
#include "docan_task_event.h"
#include "task_object_def.h"

using namespace hozon::netaos::sttask;

namespace hozon {
namespace netaos {
namespace diag {
    /**
     * @brief Class of CommandTaskBase
     *
     * This class is a command task.
     */
    class CommandTaskBase : public STCommandTask
    {
    public:
        CommandTaskBase(uint32_t commandId, STObject* parent, STObject::TaskCB callback
            , const TaskReqInfo& reqInfo, const TaskResInfo& resInfo);
        virtual ~CommandTaskBase();

        virtual TaskReqInfo&        getReqInfo();
        virtual TaskResInfo&        getResInfo();

    protected:
        virtual uint32_t     doCommand() = 0;
        virtual bool         onCommandEvent(bool isTimeout, STEvent* event);
        virtual bool         onEventAction(bool isTimeout, STEvent* event) = 0;

        TaskReqInfo         m_reqInfo;
        TaskResInfo         m_resInfo;

    private:
        CommandTaskBase(const CommandTaskBase&);
        CommandTaskBase& operator=(const CommandTaskBase&);
    };

} // end of diag
} // end of netaos
} // end of hozon
#endif  // COMMAND_TASK_BASE_H_
/* EOF */
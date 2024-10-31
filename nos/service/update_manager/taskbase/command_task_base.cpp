/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase implement
 */

#include "update_manager/taskbase/command_task_base.h"

namespace hozon {
namespace netaos {
namespace update {

CommandTaskBase::CommandTaskBase(uint32_t commandId, STObject* parent, STObject::TaskCB callback)
    : STCommandTask(commandId, parent, callback)
{

}

CommandTaskBase::~CommandTaskBase()
{
}

bool
CommandTaskBase::onCommandEvent(bool isTimeout, STEvent* event)
{
    return onEventAction(isTimeout, event);
}

} // end of update
} // end of netaos
} // end of hozon
/* EOF */
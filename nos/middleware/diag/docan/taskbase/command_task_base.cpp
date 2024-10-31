/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase implement
 */

#include "command_task_base.h"

namespace hozon {
namespace netaos {
namespace diag {

    CommandTaskBase::CommandTaskBase(uint32_t commandId, STObject* parent, STObject::TaskCB callback
        , const TaskReqInfo& reqInfo, const TaskResInfo& resInfo)
        : STCommandTask(commandId, parent, callback)
        , m_reqInfo(reqInfo)
        , m_resInfo(resInfo)
    {
        m_reqInfo.diagType  = reqInfo.diagType;
        m_reqInfo.Mtype     = reqInfo.Mtype;
        m_reqInfo.N_SA      = reqInfo.N_SA;
        m_reqInfo.N_TA      = reqInfo.N_TA;
        m_reqInfo.N_TAtype  = reqInfo.N_TAtype;
        m_reqInfo.reqBs     = reqInfo.reqBs;
        m_reqInfo.reqBsIndexExpect = reqInfo.reqBsIndexExpect;
        m_reqInfo.reqCanid  = reqInfo.reqCanid;
        m_reqInfo.reqCompletedSize = reqInfo.reqCompletedSize;
        m_reqInfo.reqContent = reqInfo.reqContent;
        m_reqInfo.reqEcu    = reqInfo.reqEcu;
        m_reqInfo.reqFs     = reqInfo.reqFs;
        m_reqInfo.reqSTmin  = reqInfo.reqSTmin;
        m_reqInfo.suppressPosRsp = reqInfo.suppressPosRsp;
    }

    CommandTaskBase::~CommandTaskBase()
    {
    }

    bool CommandTaskBase::onCommandEvent(bool isTimeout, STEvent* event)
    {
        return onEventAction(isTimeout, event);
    }

    TaskReqInfo& CommandTaskBase::getReqInfo()
    {
        return m_reqInfo;
    }

    TaskResInfo& CommandTaskBase::getResInfo()
    {
        return m_resInfo;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
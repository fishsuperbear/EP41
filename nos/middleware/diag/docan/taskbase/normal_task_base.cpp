/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class NormakTaskBase implement
 */


#include "normal_task_base.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    uint32_t NormalTaskBase::s_reqNo = 0;
    std::mutex NormalTaskBase::s_sync;

    NormalTaskBase::NormalTaskBase(uint32_t operationId,
                                    NormalTaskBase* parent,
                                    STObject::TaskCB callback,
                                    bool isTopTask)
            : STNormalTask(operationId, parent, callback, isTopTask)
            , m_reqId(0)
    {
        if (isTopTask) {
            m_reqId = generateReqId();
            DOCAN_LOG_D("generateReqId = %d!", m_reqId);
        }
        else {
            if (parent) {
                m_reqId = parent->reqId();
            }
            else {
                DOCAN_LOG_E("parent null point!");
            }
        }
    }

    NormalTaskBase::~NormalTaskBase()
    {
    }

    int32_t NormalTaskBase::generateReqId()
    {
        std::lock_guard<std::mutex> sync(s_sync);
        return (++s_reqNo > 0) ? s_reqNo : s_reqNo = 1;
    }

    int32_t NormalTaskBase::reqId()
    {
        return m_reqId;
    }

    uint32_t NormalTaskBase::doAction()
    {
        return eContinue;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */

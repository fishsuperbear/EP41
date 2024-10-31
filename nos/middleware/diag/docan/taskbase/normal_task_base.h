/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase Header
 */

#ifndef NORMAL_TASK_BASE_H_
#define NORMAL_TASK_BASE_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <vector>
#include <mutex>
#include "diag/libsttask/STNormalTask.h"
#include "diag/libsttask/STTaskContext.h"

using namespace hozon::netaos::sttask;

namespace hozon {
namespace netaos {
namespace diag {

    /**
     * @brief Class of NormalTaskBase
     *
     * This class is a normal task.
     */
    class NormalTaskBase : public STNormalTask
    {
    public:
        NormalTaskBase(uint32_t operationId, NormalTaskBase* parent, STObject::TaskCB callback, bool isTopTask);
        virtual ~NormalTaskBase();

        int32_t reqId();

    protected:
        virtual uint32_t doAction();

    private:
        int32_t generateReqId();

    private:
        int32_t m_reqId;

        static uint32_t     s_reqNo;
        static std::mutex   s_sync;

    private:
        NormalTaskBase(const NormalTaskBase&);
        NormalTaskBase& operator=(const NormalTaskBase&);
    };

} // end of diag
} // end of netaos
} // end of hozon
#endif  // NORMAL_TASK_BASE_H_
/* EOF */

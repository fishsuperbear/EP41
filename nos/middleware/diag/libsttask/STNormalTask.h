/**
 * Copyright @ 2021 - 2023 Hozon Auto Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * Hozon Auto Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

/**
 * @file  STNormalTask.h
 * @brief Class of STNormalTask
 */

#ifndef STNORMALTASK_H
#define STNORMALTASK_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "STTask.h"

namespace hozon {
namespace netaos {
namespace sttask {

    /**
     * @brief Class of STNormalTask
     *
     * TBD.
     */
    class STNormalTask : public STTask
    {
    public:
        STNormalTask(uint32_t operationId, STObject* parent, STObject::TaskCB callback, bool isTopTask);
        virtual ~STNormalTask();

    protected:
        virtual uint32_t            doAction() = 0;
        virtual void                onCallbackAction(uint32_t result);

    private:
        STNormalTask(const STNormalTask&);
        STNormalTask& operator=(const STNormalTask&);
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STNORMALTASK_H */
/* EOF */
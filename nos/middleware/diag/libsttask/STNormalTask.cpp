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
 * @file STNormalTask.cpp
 * @brief implements of STNormalTask
 */

#include "STNormalTask.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STNormalTask::STNormalTask(uint32_t operationId, STObject* parent, STObject::TaskCB callback, bool isTopTask)
        : STTask(ST_TASK_TYPE_NORMAL, operationId, parent, callback, isTopTask)
    {
    }

    STNormalTask::~STNormalTask()
    {
    }


    void STNormalTask::onCallbackAction(uint32_t result)
    {
        (void)(result);
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
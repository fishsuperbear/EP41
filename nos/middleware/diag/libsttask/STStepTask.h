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
 * @file  STStepTask.h
 * @brief Class of STStepTask
 */

#ifndef STSTEPTASK_H
#define STSTEPTASK_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "STTask.h"
#include "STLogDef.h"

namespace hozon {
namespace netaos {
namespace sttask {

    /**
     * @brief Class of STStepTask
     *
     * TBD.
     */
    class STStepTask : public STTask
    {
    public:
        STStepTask(uint32_t stepId, ST_TASK_TYPE taskType, STObject* parent, STObject::TaskCB callback);
        virtual ~STStepTask();

        bool                    onEvent(STEvent* event);
        uint32_t                getStepId() const;
        bool                    setInterrupted(uint32_t interruptReason);

        virtual std::string     toString();
        virtual std::string     getObjectName();

    protected:
        virtual bool            onStepEvent(bool isTimeout, STEvent* event);
        virtual uint32_t        doAction() = 0;
        virtual void            onCallbackAction(uint32_t result);
        virtual void            onInterruptAction(uint32_t interruptReason);
        virtual bool            checkOnIntterupt(uint32_t interruptReason);

        bool                    restartTimer(uint32_t timeout);

    private:
        const uint32_t          m_stepId;
        uint32_t                m_scheduledEventId;

    private:
        STStepTask(const STStepTask&);
        STStepTask& operator=(const STStepTask&);
    };

} // end of sttask
} // end of netaos
} // end of hozon
#endif /* STSTEPTASK_H */
/* EOF */
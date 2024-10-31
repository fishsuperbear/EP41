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
 * @file  STTimerTask.h
 * @brief Class of STTimerTask
 */

#ifndef STTIMERTASK_H
#define STTIMERTASK_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "STStepTask.h"

namespace hozon {
namespace netaos {
namespace sttask {

    /**
     * @brief Class of STTimerTask
     *
     * TBD.
     */
    class STTimerTask : public STStepTask
    {
    public:
        STTimerTask(uint32_t timerId, STObject* parent, STObject::TaskCB callback, uint32_t timeoutInMs);
        virtual ~STTimerTask();
        bool                isWaitEvent() const;

        virtual std::string toString();

    protected:
        virtual bool        onStepEvent(bool isTimeout, STEvent* event);
        virtual uint32_t    doAction();
        virtual bool        onTimerEvent(bool isTimeout, STEvent* event) = 0;
        virtual bool        onTimerInterruptCheck(uint32_t interruptReason);
        virtual bool        checkOnIntterupt(uint32_t interruptReason);

    private:
        uint32_t            m_timeout;

    private:
        STTimerTask(const STTimerTask&);
        STTimerTask& operator=(const STTimerTask&);
    };

} // end of sttask
} // end of netaos
} // end of hozon
#endif /* STTIMERTASK_H */
/* EOF */
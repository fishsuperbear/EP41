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
 * @file  STCommandTask.h
 * @brief Class of STCommandTask
 */

#ifndef STCOMMANDTASK_H
#define STCOMMANDTASK_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "STStepTask.h"

namespace hozon {
namespace netaos {
namespace sttask {

    class STEvent;
    /**
     * @brief Class of STCommandTask
     *
     * This class is a command task.
     */
    class STCommandTask : public STStepTask
    {
    public:
        STCommandTask(uint32_t commandId, STObject* parent, STObject::TaskCB callback);
        virtual ~STCommandTask();

        virtual void            setTaskResult(uint32_t result);
        bool                    isWaitEvent() const;
        bool                    isExecuted() const;

        virtual std::string     toString();
        virtual std::string     getObjectName();

    protected:
        virtual bool        onStepEvent(bool isTimeout, STEvent* event);
        virtual uint32_t    doAction();
        virtual uint32_t    doCommand() = 0;
        virtual bool        onCommandEvent(bool isTimeout, STEvent* event) = 0;
        virtual bool        onCommandInterruptCheck(uint32_t interruptReason);
        bool                waitEvent(uint32_t timeout);
        virtual bool        checkOnIntterupt(uint32_t interruptReason);

    private:
        enum COMMAND_STATUS
        {
            COMMAND_STATUS_NOTSTARTED,
            COMMAND_STATUS_EXECUTED,
            COMMAND_STATUS_WAIT,
            COMMAND_STATUS_FINISHED,
        };

        uint32_t                m_commandId;
        COMMAND_STATUS          m_commandStatus;

    private:
        STCommandTask(const STCommandTask&);
        STCommandTask& operator=(const STCommandTask&);
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STCOMMANDTASK_H */
/* EOF */
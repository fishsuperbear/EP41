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
 * @file  STModuleBase.h
 * @brief Class of STModuleBase
 */

#ifndef STMODULEBASE_H
#define STMODULEBASE_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "STObject.h"

namespace hozon {
namespace netaos {
namespace sttask {

    class STStepTask;
    class STNormalTask;
    class STEvent;
    class STCall;
    class STTaskContext;
    class STModuleManager;
    /**
     * @brief Class of STModuleBase
     *
     * This class represent a module.
     */
    class STModuleBase : public STObject
    {
    public:
        explicit STModuleBase(uint32_t moduleID);
        virtual ~STModuleBase();

        uint32_t                getModuleID() const;

        virtual void            onOperationPost(uint32_t operationId, STNormalTask* topTask);
        virtual uint32_t        onOperationStart(uint32_t operationId, STNormalTask* topTask);
        virtual void            onOperationStarted(uint32_t operationId, STNormalTask* topTask);
        virtual void            onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* topTask);
        virtual void            onOperationInterrupt(uint32_t operationId, uint32_t interruptReason, STNormalTask* topTask);

        virtual void            onStepPost(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual uint32_t        onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual void            onStepStarted(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual void            onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask);
        virtual void            onStepInterrupt(uint32_t operationId, uint32_t stepId, uint32_t interruptReason, STStepTask* stepTask);

        virtual void            onEvent(uint32_t eventKind, uint32_t eventId, STEvent* event);
        virtual void            onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event);
        virtual void            onCall(uint32_t callKind, uint32_t callId, STCall* callObj);

        virtual std::string     toString();

    protected:
        uint32_t                post(STTask* task);
        void                    post(STEvent* event);
        void                    call(STCall* callObj);
        void                    setMonitorCB(STObject* target, STObject::MonitorCB monitorCB);
        void                    unsetMonitorCB(STObject* target);
        uint32_t                startEventWatcher(STObject* target, STObject::EventWatcherCB watcherCB, uint32_t timeout = ST_TIME_INFINITE);
        void                    stopEventWatcher(uint32_t evtWatcherId);
        uint32_t                startPeriod(STObject* target, STObject::PeriodCB periodCB, uint32_t periodTime);
        void                    stopPeriod(uint32_t periodId);
        uint32_t                scheduleEvent(const uint16_t timeout);
        void                    unscheduleEvent(const uint32_t id);

        bool                    interruptOperation(uint32_t operationId, uint32_t interruptReason);
        bool                    interruptCommand(uint32_t commandId, uint32_t interruptReason);
        STTaskContext*          getContext();
        STModuleManager*        getModuleManager();

    private:
        STModuleBase(const STModuleBase&);
        STModuleBase& operator=(const STModuleBase&);
        const uint32_t            m_moduleID;
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STMODULEBASE_H */
/* EOF */
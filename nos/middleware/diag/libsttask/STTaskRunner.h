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
 * @file  STTaskRunner.h
 * @brief Class of STTaskRunner
 */

#ifndef STTASKRUNNER_H
#define STTASKRUNNER_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif


#include "STObject.h"
#include "STTaskThread.h"
#include "STTaskConfig.h"
#include "STTaskContext.h"
#include "STModuleManager.h"

namespace hozon {
namespace netaos {
namespace sttask {

    class STTask;
    class STEvent;
    class STCall;
    class STNormalTask;
    class STStepTask;
    /**
     * @brief Class of STTaskRunner
     *
     * TBD.
     */
    class STTaskRunner : public STObject
    {
    public:
        STTaskRunner();
        virtual ~STTaskRunner();

        void    init();
        void    start();
        void    stop();
        void    deinit();

        virtual const char* getTaskThreadName() = 0;

        STTaskThread*       getTaskThread();
        STTaskConfig*       getConfiguration();
        STTaskContext*      getContext();
        STModuleManager*    getModuleManager();


        uint32_t            post(STTask* task);
        void                post(STEvent* event);
        void                call(STCall* callObj);
        void                setMonitorCB(STObject* target, STObject::MonitorCB monitorCB);
        void                unsetMonitorCB(STObject* target);

        uint32_t            startEventWatcher(STObject* target, STObject::EventWatcherCB watcherCB, uint32_t timeout = ST_TIME_INFINITE);
        void                stopEventWatcher(uint32_t evtWatcherId);
        bool                isEventWatcherAlive(uint32_t evtWatcherId);

        uint32_t            startPeriod(STObject* target, STObject::PeriodCB periodCB, uint32_t periodTime);
        void                stopPeriod(uint32_t periodId);
        bool                isPeriodAlive(uint32_t periodId);

        uint32_t            scheduleEvent(const uint32_t timeout);
        void                unscheduleEvent(const uint32_t id);
        uint32_t            nextId();

        bool                interruptOperation(uint32_t operationId, uint32_t interruptReason);
        bool                interruptCommand(uint32_t commandId, uint32_t interruptReason);

    protected:
        virtual void        onInit();
        virtual void        onStart();
        virtual void        onStop();
        virtual void        onStopped();
        virtual void        onDeinit();
        virtual void        onOperationPost(uint32_t operationId, STNormalTask* topTask);
        virtual uint32_t    onOperationStart(uint32_t operationId, STNormalTask* topTask);
        virtual void        onOperationStarted(uint32_t operationId, STNormalTask* topTask);
        virtual void        onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* topTask);
        virtual void        onOperationInterrupt(uint32_t operationId, uint32_t interruptReason, STNormalTask* topTask);

        virtual void        onStepPost(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual uint32_t    onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual void        onStepStarted(uint32_t operationId, uint32_t stepId, STStepTask* stepTask);
        virtual void        onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask);
        virtual void        onStepInterrupt(uint32_t operationId, uint32_t stepId, uint32_t interruptReason, STStepTask* stepTask);

        virtual void        onEvent(uint32_t eventKind, uint32_t eventId, STEvent* event);
        virtual void        onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event);
        virtual void        onCall(uint32_t callKind, uint32_t callId, STCall* callObj);

    private:
        void                onTaskPost(STTask* task);
        uint32_t            onTaskStart(STTask* task);
        void                onTaskEnd(STTask* task, uint32_t result);
        void                onTaskDelete(STTask* task);
        void                onTaskInterrupt(STTask* task, uint32_t interruptReason);
        friend              class STTask;

        uint32_t            checkNormalTaskPost(STNormalTask* task);
        uint32_t            checkStepTaskPost(STStepTask* stepTask);

        bool                checkOperationExecute(STNormalTask* task);
        bool                checkStepTaskExecute(STStepTask* stepTask);

        void                onMonitorEvent(STEvent* event);
        void                onMonitorUnexpectedEvent(STEvent* event);
        friend              class STTaskThread;

        void                doOnStepPost(STStepTask* stepTask);
        void                doOnOperationPost(STNormalTask* opTask);

        uint32_t            doOnStepStart(STStepTask* stepTask);
        uint32_t            doOnOperationStart(STNormalTask* opTask);

        void                doOnStepEnd(STStepTask* stepTask, uint32_t result);
        void                doOnOperationEnd(STNormalTask* opTask, uint32_t result);

        void                doOnStepInterrupt(STStepTask* stepTask, uint32_t interruptReason);
        void                doOnOperationInterrupt(STNormalTask* opTask, uint32_t interruptReason);

    private:
        STTaskThread        m_taskThread;
        STTaskConfig        m_configuration;
        STTaskContext       m_context;
        STModuleManager     m_moduleManager;

    private:
        STTaskRunner(const STTaskRunner&);
        STTaskRunner& operator=(const STTaskRunner&);

    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STTASKRUNNER_H */
/* EOF */
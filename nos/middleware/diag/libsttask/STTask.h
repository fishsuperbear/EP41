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
 * @file  STTask.h
 * @brief Class of STTask
 */
#ifndef STTASK_H
#define STTASK_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "STObject.h"
#include "STObjectDef.h"

#define CAST_COMMANDTASK(t) (static_cast<STCommandTask*>(t))
#define CAST_TIMERTASK(t) (static_cast<STTimerTask*>(t))
#define CAST_STEPTASK(t) (static_cast<STStepTask*>(t))
#define CAST_NORMALTASK(t) (static_cast<STNormalTask*>(t))
#define CAST_TASK(t) (static_cast<STTask*>(t))

namespace hozon {
namespace netaos {
namespace sttask {

    class STEvent;
    class STTaskRunner;
    class STTaskContext;
    class STModuleManager;
    /**
     * @brief Class of STTask
     *
     * This class is a normal task.
     */
    class STTask : public STObject
    {
    public:
        STTask(ST_TASK_TYPE taskType, uint32_t operationId, STObject* parent, STObject::TaskCB callback, bool isTopTask);
        virtual ~STTask();

        ST_TASK_TYPE        getTaskType() const;
        uint32_t            getOperationId() const;
        bool                isTopTask() const;
        bool                isNormalTask() const;
        bool                isTimerTask() const;
        bool                isCommandTask() const;
        bool                isStepTask() const;

        uint32_t            getTaskResult() const;
        virtual void        setTaskResult(uint32_t result);
        void                onCallbackResult(uint32_t result);

        void                onPost(STTaskRunner* runner);
        uint32_t            doTask();
        void                deleteTask();

        STTask*             getTopTask();
        STObject*           getParent() const;

        bool                isIntterrupted() const;
        uint32_t            getIntteruptedReason() const;

        virtual std::string toString();
        virtual std::string getObjectName();

    protected:
        uint32_t            post(STTask* task);
        void                post(STEvent* event);
        void                call(STCall* callObj);
        void                setMonitorCB(STObject* target, STObject::MonitorCB monitorCB);
        void                unsetMonitorCB(STObject* target);
        uint32_t            startEventWatcher(STObject* target, STObject::EventWatcherCB watcherCB, uint32_t timeout = ST_TIME_INFINITE);
        void                stopEventWatcher(uint32_t evtWatcherId);
        uint32_t            startPeriod(STObject* target, STObject::PeriodCB periodCB, uint32_t periodTime);
        void                stopPeriod(uint32_t periodId);
        uint32_t            scheduleEvent(const uint32_t timeout);
        void                unscheduleEvent(const uint32_t id);

        STTaskContext*      getContext();
        STModuleManager*    getModuleManager();


        virtual uint32_t    doAction() = 0;
        virtual void        onInterruptAction(uint32_t interruptReason);
        virtual void        onCallbackAction(uint32_t result) = 0;

        void                onInterrupt(uint32_t interruptReason);
        void                onTaskPost();
        uint32_t            onTaskStart();
        void                onTaskEnd(uint32_t result);


        ST_TASK_TYPE        m_taskType;
        uint32_t            m_operationId;
        bool                m_isTopTask;
        STObject*           m_parent;
        STObject::TaskCB    m_taskCallback;
        uint32_t            m_taskResult;
        uint32_t            m_interruptReason;

    private:
        STTask(const STTask&);
        STTask& operator=(const STTask&);
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STTASK_H */
/* EOF */
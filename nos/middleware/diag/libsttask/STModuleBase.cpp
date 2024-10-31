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
 * @file STModuleBase.cpp
 * @brief Implements of STModuleBase class
 */

#include "STModuleBase.h"
#include "STTaskRunner.h"
#include "STLogDef.h"
#include "STCommandTask.h"
#include "STTimerTask.h"
#include "STNormalTask.h"
#include "STEvent.h"
#include "STCall.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STModuleBase::STModuleBase(uint32_t moduleID)
        : STObject(ST_OBJECT_TYPE_MODULE)
        , m_moduleID(moduleID)
    {

    }


    STModuleBase::~STModuleBase()
    {

    }

    uint32_t STModuleBase::getModuleID() const
    {
        return m_moduleID;
    }

    void STModuleBase::onOperationPost(uint32_t operationId, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(topTask);
    }

    uint32_t STModuleBase::onOperationStart(uint32_t operationId, STNormalTask* topTask)
    {
        STLOG_D("onOperationStart()");
        (void)(operationId);
        (void)(topTask);
        return eContinue;
    }

    void STModuleBase::onOperationStarted(uint32_t operationId, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(topTask);
    }

    void STModuleBase::onOperationEnd(uint32_t operationId, uint32_t result, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(result);
        (void)(topTask);
    }

    void STModuleBase::onOperationInterrupt(uint32_t operationId, uint32_t interruptReason, STNormalTask* topTask)
    {
        (void)(operationId);
        (void)(interruptReason);
        (void)(topTask);
    }

    void STModuleBase::onStepPost(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(stepTask);
    }

    uint32_t STModuleBase::onStepStart(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        STLOG_D("doOnStepStart()");
        (void)(operationId);
        (void)(stepId);
        (void)(stepTask);
        return eContinue;
    }

    void STModuleBase::onStepStarted(uint32_t operationId, uint32_t stepId, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(stepTask);
    }

    void STModuleBase::onStepEnd(uint32_t operationId, uint32_t stepId, uint32_t result, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(result);
        (void)(stepTask);
    }

    void STModuleBase::onStepInterrupt(uint32_t operationId, uint32_t stepId, uint32_t interruptReason, STStepTask* stepTask)
    {
        (void)(operationId);
        (void)(stepId);
        (void)(interruptReason);
        (void)(stepTask);
    }

    void STModuleBase::onEvent(uint32_t eventKind, uint32_t eventId, STEvent* event)
    {
        (void)(eventKind);
        (void)(eventId);
        (void)(event);
    }

    void STModuleBase::onUnexpectedEvent(uint32_t eventKind, uint32_t eventId, STEvent* event)
    {
        (void)(eventKind);
        (void)(eventId);
        (void)(event);
    }

    void STModuleBase::onCall(uint32_t callKind, uint32_t callId, STCall* callObj)
    {
        (void)(callKind);
        (void)(callId);
        (void)(callObj);
    }

    std::string STModuleBase::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), ("Module[%p, %d]"), this, getModuleID());
        val.assign(buf, strlen(buf));
        return val;
    }

    uint32_t STModuleBase::post(STTask* task)
    {
        if (nullptr != task) {
            if (getTaskRunner()) {
                return getTaskRunner()->post(task);
            }
            delete task;
            task = nullptr;
        }

        return eMemErr;
    }

    void STModuleBase::post(STEvent* event)
    {
        if (nullptr != event) {
            if (getTaskRunner()) {
                getTaskRunner()->post(event);
                return;
            }
            delete event;
            event = nullptr;
        }
    }

    void STModuleBase::call(STCall* call)
    {
        if (nullptr != call) {
            if (getTaskRunner()) {
                getTaskRunner()->call(call);
                return;
            }
            delete call;
            call = nullptr;
        }
    }

    void STModuleBase::setMonitorCB(STObject* target, STObject::MonitorCB monitorCB)
    {
        if (getTaskRunner()) {
            getTaskRunner()->setMonitorCB(target, monitorCB);
            return;
        }
    }

    void STModuleBase::unsetMonitorCB(STObject* target)
    {
        if (getTaskRunner()) {
            getTaskRunner()->unsetMonitorCB(target);
            return;
        }
    }

    uint32_t STModuleBase::startEventWatcher(STObject* target, STObject::EventWatcherCB watcherCB, uint32_t timeout)
    {
        if (getTaskRunner()) {
            return getTaskRunner()->startEventWatcher(target, watcherCB, timeout);
        }
        return 0;
    }

    void STModuleBase::stopEventWatcher(uint32_t evtWatcherId)
    {
        if (getTaskRunner()) {
            getTaskRunner()->stopEventWatcher(evtWatcherId);
            return;
        }
    }

    uint32_t STModuleBase::startPeriod(STObject* target, STObject::PeriodCB periodCB, uint32_t periodTime)
    {
        if (getTaskRunner()) {
            return getTaskRunner()->startPeriod(target, periodCB, periodTime);
        }
        return 0;
    }

    void STModuleBase::stopPeriod(uint32_t periodId)
    {
        if (getTaskRunner()) {
            getTaskRunner()->stopPeriod(periodId);
            return;
        }
    }

    uint32_t STModuleBase::scheduleEvent(const uint16_t timeout)
    {
        if (getTaskRunner()) {
            return getTaskRunner()->scheduleEvent(timeout);
        }
        return 0;
    }

    void STModuleBase::unscheduleEvent(const uint32_t id)
    {
        if (getTaskRunner()) {
            getTaskRunner()->unscheduleEvent(id);
            return;
        }
    }

    bool STModuleBase::interruptOperation(uint32_t operationId, uint32_t interruptReason)
    {
        if (getTaskRunner()) {
            return getTaskRunner()->interruptOperation(operationId, interruptReason);
        }
        return false;
    }

    bool STModuleBase::interruptCommand(uint32_t commandId, uint32_t interruptReason)
    {
        if (getTaskRunner()) {
            return getTaskRunner()->interruptCommand(commandId, interruptReason);
        }
        return false;
    }

    STTaskContext* STModuleBase::getContext()
    {
        if (getTaskRunner()) {
            return getTaskRunner()->getContext();
        }
        return nullptr;
    }

    STModuleManager* STModuleBase::getModuleManager()
    {
        if (getTaskRunner()) {
            return getTaskRunner()->getModuleManager();
        }
        return nullptr;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */

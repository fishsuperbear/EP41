/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateTaskEvent implement
 */


#include "update_manager/taskbase/update_task_event.h"

namespace hozon {
namespace netaos {
namespace update {

    UpdateTaskEvent::UpdateTaskEvent(uint32_t evtKind, uint32_t evtId, int32_t evtVal1, int32_t evtVal2,  const std::vector<uint8_t>& evtData)
        : STEvent(evtKind, evtId)
        , m_evtVal1(evtVal1)
        , m_evtVal2(evtVal2)
    {
        m_evtData = evtData;
    }

    UpdateTaskEvent::~UpdateTaskEvent()
    {
        m_evtData.clear();
    }

    bool UpdateTaskEvent::checkEvent(uint32_t evtKind, STEvent* event)
    {
        if (nullptr == event) {
            return false;
        }

        if (evtKind == event->getEventKind()) {
            return true;
        }

        return false;
    }

    bool UpdateTaskEvent::checkEvent(uint32_t evtKind, uint32_t evtId, STEvent* event)
    {
        if (nullptr == event) {
            return false;
        }

        UpdateTaskEvent* evt = static_cast<UpdateTaskEvent*>(event);
        if (nullptr == evt) {
            return false;
        }

        if (evtKind == evt->getEventKind() && evtId == evt->getEventId()) {
            return true;
        }

        return false;
    }

    bool UpdateTaskEvent::checkEvent(uint32_t evtKind, uint32_t evtId, int32_t evtVal1, STEvent* event)
    {
        if (nullptr == event) {
            return false;
        }

        UpdateTaskEvent* evt = static_cast<UpdateTaskEvent*>(event);
        if (nullptr == evt) {
            return false;
        }

        if (evtKind == evt->getEventKind() && evtId == evt->getEventId()
            && evtVal1 == evt->getEvtVal1()) {
            return true;
        }

        return false;
    }

    bool UpdateTaskEvent::checkEvent(uint32_t evtKind, uint32_t evtId, int32_t evtVal1, int32_t evtVal2, STEvent* event)
    {
        if (nullptr == event) {
            return false;
        }

        UpdateTaskEvent* evt = static_cast<UpdateTaskEvent*>(event);
        if (nullptr == evt) {
            return false;
        }

        if (evtKind == evt->getEventKind() && evtId == evt->getEventId()
            && evtVal1 == evt->getEvtVal1() && evtVal2 == evt->getEvtVal2()) {
            return true;
        }

        return false;
    }

    int32_t UpdateTaskEvent::getEvtVal1() const
    {
        return m_evtVal1;
    }

    int32_t UpdateTaskEvent::getEvtVal2() const
    {
        return m_evtVal2;
    }

    std::vector<uint8_t>& UpdateTaskEvent::getEvtData()
    {
        return m_evtData;
    }

    void UpdateTaskEvent::setEvtData(const std::vector<uint8_t>& data)
    {
        m_evtData = data;
    }

    uint32_t UpdateTaskEvent::getEcu() const
    {
        return getEventId();
    }

    uint16_t UpdateTaskEvent::getCanid() const
    {
        return m_evtVal1;
    }

    uint8_t UpdateTaskEvent::getFrameType() const
    {
        return m_evtVal2;
    }

} // end of update
} // end of netaos
} // end of hozon
/* EOF */
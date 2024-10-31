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
 * @file STEvent.cpp
 * @brief implements of STEvent
 */


#include "STEvent.h"
#include <string.h>

namespace hozon {
namespace netaos {
namespace sttask {

    bool STEvent::isTimerEvent(uint32_t evtId, STEvent* evt)
    {
        if (nullptr == evt) {
            return false;
        }
        if (evt->getEventKind() == eEventKind_TimerEvent
            && evt->getEventId() == evtId) {
            return true;
        }

        return false;
    }

    STEvent::STEvent(uint32_t evtKind, uint32_t evtId)
        : m_evtKind(evtKind)
        , m_evtId(evtId)
    {
    }

    STEvent::~STEvent()
    {
    }

    uint32_t STEvent::getEventKind() const
    {
        return m_evtKind;
    }

    uint32_t STEvent::getEventId() const
    {
        return m_evtId;
    }

    std::string STEvent::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), "Event[%p, %X, %X]", this, getEventKind(), getEventId());
        val.assign(buf, strlen(buf));
        return val;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */
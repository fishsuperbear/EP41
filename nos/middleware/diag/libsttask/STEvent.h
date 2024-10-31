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
 * @file  STEvent.h
 * @brief Class of STEvent
 */
#ifndef STEVENT_H
#define STEVENT_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <string>
#include "STObjectDef.h"

namespace hozon {
namespace netaos {
namespace sttask {

    /**
     * @brief Class of STEvent
     *
     * This class is a normal task.
     */
    class STEvent
    {
    public:
        static bool isTimerEvent(uint32_t evtId, STEvent* evt);

    public:
        STEvent(uint32_t evtKind, uint32_t evtId);
        virtual ~STEvent();

        uint32_t              getEventKind() const;
        uint32_t              getEventId() const;

        virtual std::string    toString();

    private:
        STEvent(const STEvent&);
        STEvent& operator=(const STEvent&);

        const uint32_t    m_evtKind;
        const uint32_t    m_evtId;
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STEVENT_H */
/* EOF */
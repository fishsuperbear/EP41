/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateTaskEvent Header
 */

#ifndef TASK_EVENT_BASE_H_
#define TASK_EVENT_BASE_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif


#include <vector>
#include "diag/libsttask/STEvent.h"
#include "update_manager/taskbase/task_object_def.h"

using namespace hozon::netaos::sttask;

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class UpdateTaskEvent
     *
     */
    class UpdateTaskEvent : public STEvent
    {
    public:
        UpdateTaskEvent(uint32_t evtKind, uint32_t evtId, int32_t evtVal1, int32_t evtVal2, const std::vector<uint8_t>& evtData);
        virtual ~UpdateTaskEvent();

        static bool checkEvent(uint32_t evtKind, STEvent* event);
        static bool checkEvent(uint32_t evtKind, uint32_t evtId, STEvent* event);
        static bool checkEvent(uint32_t evtKind, uint32_t evtId, int32_t evtVal1, STEvent* event);
        static bool checkEvent(uint32_t evtKind, uint32_t evtId, int32_t evtVal1, int32_t evtVal2, STEvent* event);

        int32_t getEvtVal1() const;
        int32_t getEvtVal2() const;
        std::vector<uint8_t>& getEvtData();
        void setEvtData(const std::vector<uint8_t>& data);

        uint32_t getEcu() const;
        uint16_t getCanid() const;
        uint8_t getFrameType() const;

    private:
        UpdateTaskEvent(const UpdateTaskEvent&);
        UpdateTaskEvent& operator=(const UpdateTaskEvent&);

    private:
        int32_t                 m_evtVal1;
        int32_t                 m_evtVal2;
        std::vector<uint8_t>    m_evtData;
    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // TASK_EVENT_BASE_H_
/* EOF */
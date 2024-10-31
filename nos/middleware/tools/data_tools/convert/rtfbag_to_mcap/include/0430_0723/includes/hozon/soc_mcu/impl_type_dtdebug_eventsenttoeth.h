/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_EVENTSENTTOETH_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_EVENTSENTTOETH_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_EventSentToETH {
    ::UInt16 eventSentId;
    ::UInt8 eventSentObj;
    ::UInt8 eventSentToETHStatus;
    ::UInt16 reportCnt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(eventSentId);
        fun(eventSentObj);
        fun(eventSentToETHStatus);
        fun(reportCnt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(eventSentId);
        fun(eventSentObj);
        fun(eventSentToETHStatus);
        fun(reportCnt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("eventSentId", eventSentId);
        fun("eventSentObj", eventSentObj);
        fun("eventSentToETHStatus", eventSentToETHStatus);
        fun("reportCnt", reportCnt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("eventSentId", eventSentId);
        fun("eventSentObj", eventSentObj);
        fun("eventSentToETHStatus", eventSentToETHStatus);
        fun("reportCnt", reportCnt);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_EventSentToETH& t) const
    {
        return (eventSentId == t.eventSentId) && (eventSentObj == t.eventSentObj) && (eventSentToETHStatus == t.eventSentToETHStatus) && (reportCnt == t.reportCnt);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_EVENTSENTTOETH_H

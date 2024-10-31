/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_VEHICLE_DATA_COLLECT_IMPL_TYPE_TRIGGERSIGNALFRAME_H
#define HOZON_VEHICLE_DATA_COLLECT_IMPL_TYPE_TRIGGERSIGNALFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/vehicle_data_collect/impl_type_triggersignalvector.h"
#include "impl_type_boolean.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace vehicle_data_collect {
struct TriggerSignalFrame {
    ::hozon::common::CommonHeader header;
    ::hozon::vehicle_data_collect::TriggerSignalVector triggerSignals;
    ::Boolean isValid;
    ::UInt8 signalType;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(triggerSignals);
        fun(isValid);
        fun(signalType);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(triggerSignals);
        fun(isValid);
        fun(signalType);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("triggerSignals", triggerSignals);
        fun("isValid", isValid);
        fun("signalType", signalType);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("triggerSignals", triggerSignals);
        fun("isValid", isValid);
        fun("signalType", signalType);
    }

    bool operator==(const ::hozon::vehicle_data_collect::TriggerSignalFrame& t) const
    {
        return (header == t.header) && (triggerSignals == t.triggerSignals) && (isValid == t.isValid) && (signalType == t.signalType);
    }
};
} // namespace vehicle_data_collect
} // namespace hozon


#endif // HOZON_VEHICLE_DATA_COLLECT_IMPL_TYPE_TRIGGERSIGNALFRAME_H

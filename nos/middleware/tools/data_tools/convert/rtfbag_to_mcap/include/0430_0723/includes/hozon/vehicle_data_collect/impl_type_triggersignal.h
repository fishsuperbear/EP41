/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_VEHICLE_DATA_COLLECT_IMPL_TYPE_TRIGGERSIGNAL_H
#define HOZON_VEHICLE_DATA_COLLECT_IMPL_TYPE_TRIGGERSIGNAL_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "hozon/statemachine/impl_type_datavalue.h"

namespace hozon {
namespace vehicle_data_collect {
struct TriggerSignal {
    ::UInt8 data_type;
    ::String param_name;
    ::hozon::statemachine::DataValue param_value;
    ::UInt8 reserved;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(data_type);
        fun(param_name);
        fun(param_value);
        fun(reserved);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(data_type);
        fun(param_name);
        fun(param_value);
        fun(reserved);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("data_type", data_type);
        fun("param_name", param_name);
        fun("param_value", param_value);
        fun("reserved", reserved);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("data_type", data_type);
        fun("param_name", param_name);
        fun("param_value", param_value);
        fun("reserved", reserved);
    }

    bool operator==(const ::hozon::vehicle_data_collect::TriggerSignal& t) const
    {
        return (data_type == t.data_type) && (param_name == t.param_name) && (param_value == t.param_value) && (reserved == t.reserved);
    }
};
} // namespace vehicle_data_collect
} // namespace hozon


#endif // HOZON_VEHICLE_DATA_COLLECT_IMPL_TYPE_TRIGGERSIGNAL_H

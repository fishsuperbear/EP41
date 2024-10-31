/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGEGOHMIFRAME_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGEGOHMIFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "hozon/chassis/impl_type_algegowarninginfo.h"
#include "hozon/chassis/impl_type_algegoparkhmiinfo.h"

namespace hozon {
namespace chassis {
struct AlgEgoHmiFrame {
    ::hozon::common::CommonHeader header;
    bool isValid;
    ::hozon::chassis::AlgEgoWarningInfo warnning_info;
    ::hozon::chassis::AlgEgoParkHmiInfo park_hmi_info;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(isValid);
        fun(warnning_info);
        fun(park_hmi_info);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(isValid);
        fun(warnning_info);
        fun(park_hmi_info);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("isValid", isValid);
        fun("warnning_info", warnning_info);
        fun("park_hmi_info", park_hmi_info);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("isValid", isValid);
        fun("warnning_info", warnning_info);
        fun("park_hmi_info", park_hmi_info);
    }

    bool operator==(const ::hozon::chassis::AlgEgoHmiFrame& t) const
    {
        return (header == t.header) && (isValid == t.isValid) && (warnning_info == t.warnning_info) && (park_hmi_info == t.park_hmi_info);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGEGOHMIFRAME_H

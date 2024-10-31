/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_SOFTWAREVERSIONITEM_H
#define MDC_SWM_IMPL_TYPE_SOFTWAREVERSIONITEM_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_int32.h"

namespace mdc {
namespace swm {
struct SoftwareVersionItem {
    ::String deviceName;
    ::Int32 swName;
    ::String version;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(deviceName);
        fun(swName);
        fun(version);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(deviceName);
        fun(swName);
        fun(version);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("deviceName", deviceName);
        fun("swName", swName);
        fun("version", version);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("deviceName", deviceName);
        fun("swName", swName);
        fun("version", version);
    }

    bool operator==(const ::mdc::swm::SoftwareVersionItem& t) const
    {
        return (deviceName == t.deviceName) && (swName == t.swName) && (version == t.version);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_SOFTWAREVERSIONITEM_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_VERSIONITEM_H
#define MDC_SWM_IMPL_TYPE_VERSIONITEM_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_int8.h"
#include "mdc/swm/impl_type_versioninfovector.h"
#include "impl_type_uint8.h"

namespace mdc {
namespace swm {
struct VersionItem {
    ::String name;
    ::String location;
    ::Int8 state;
    ::Int8 type;
    ::mdc::swm::VersionInfoVector versions;
    ::UInt8 deviceStatus;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(name);
        fun(location);
        fun(state);
        fun(type);
        fun(versions);
        fun(deviceStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(name);
        fun(location);
        fun(state);
        fun(type);
        fun(versions);
        fun(deviceStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("name", name);
        fun("location", location);
        fun("state", state);
        fun("type", type);
        fun("versions", versions);
        fun("deviceStatus", deviceStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("name", name);
        fun("location", location);
        fun("state", state);
        fun("type", type);
        fun("versions", versions);
        fun("deviceStatus", deviceStatus);
    }

    bool operator==(const ::mdc::swm::VersionItem& t) const
    {
        return (name == t.name) && (location == t.location) && (state == t.state) && (type == t.type) && (versions == t.versions) && (deviceStatus == t.deviceStatus);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_VERSIONITEM_H

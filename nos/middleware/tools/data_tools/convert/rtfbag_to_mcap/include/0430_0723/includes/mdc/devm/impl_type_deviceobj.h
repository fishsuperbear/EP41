/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_DEVICEOBJ_H
#define MDC_DEVM_IMPL_TYPE_DEVICEOBJ_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"

namespace mdc {
namespace devm {
struct DeviceObj {
    ::String deviceName;
    ::UInt32 deviceId;
    ::UInt8 deviceType;
    ::UInt8 upgradable;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(deviceName);
        fun(deviceId);
        fun(deviceType);
        fun(upgradable);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(deviceName);
        fun(deviceId);
        fun(deviceType);
        fun(upgradable);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("deviceName", deviceName);
        fun("deviceId", deviceId);
        fun("deviceType", deviceType);
        fun("upgradable", upgradable);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("deviceName", deviceName);
        fun("deviceId", deviceId);
        fun("deviceType", deviceType);
        fun("upgradable", upgradable);
    }

    bool operator==(const ::mdc::devm::DeviceObj& t) const
    {
        return (deviceName == t.deviceName) && (deviceId == t.deviceId) && (deviceType == t.deviceType) && (upgradable == t.upgradable);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_DEVICEOBJ_H

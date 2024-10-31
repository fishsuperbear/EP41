/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_UPGRADEINFO_H
#define MDC_DEVM_IMPL_TYPE_UPGRADEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "mdc/devm/impl_type_updateinfolist.h"

namespace mdc {
namespace devm {
struct UpgradeInfo {
    ::String deviceName;
    ::UInt32 deviceId;
    ::UInt8 deviceType;
    ::UInt8 upgradable;
    ::mdc::devm::UpdateInfoList updateList;

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
        fun(updateList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(deviceName);
        fun(deviceId);
        fun(deviceType);
        fun(upgradable);
        fun(updateList);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("deviceName", deviceName);
        fun("deviceId", deviceId);
        fun("deviceType", deviceType);
        fun("upgradable", upgradable);
        fun("updateList", updateList);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("deviceName", deviceName);
        fun("deviceId", deviceId);
        fun("deviceType", deviceType);
        fun("upgradable", upgradable);
        fun("updateList", updateList);
    }

    bool operator==(const ::mdc::devm::UpgradeInfo& t) const
    {
        return (deviceName == t.deviceName) && (deviceId == t.deviceId) && (deviceType == t.deviceType) && (upgradable == t.upgradable) && (updateList == t.updateList);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_UPGRADEINFO_H

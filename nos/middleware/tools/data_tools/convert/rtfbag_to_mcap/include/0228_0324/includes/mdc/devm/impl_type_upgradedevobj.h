/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_UPGRADEDEVOBJ_H
#define MDC_DEVM_IMPL_TYPE_UPGRADEDEVOBJ_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint8.h"
#include "mdc/devm/impl_type_updateinfolist.h"

namespace mdc {
namespace devm {
struct UpgradeDevObj {
    ::String deviceName;
    ::UInt8 deviceStatus;
    ::mdc::devm::UpdateInfoList upgradeList;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(deviceName);
        fun(deviceStatus);
        fun(upgradeList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(deviceName);
        fun(deviceStatus);
        fun(upgradeList);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("deviceName", deviceName);
        fun("deviceStatus", deviceStatus);
        fun("upgradeList", upgradeList);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("deviceName", deviceName);
        fun("deviceStatus", deviceStatus);
        fun("upgradeList", upgradeList);
    }

    bool operator==(const ::mdc::devm::UpgradeDevObj& t) const
    {
        return (deviceName == t.deviceName) && (deviceStatus == t.deviceStatus) && (upgradeList == t.upgradeList);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_UPGRADEDEVOBJ_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_IMPL_TYPE_UPDATEINFO_H
#define MDC_DEVM_IMPL_TYPE_UPDATEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_uint8.h"

namespace mdc {
namespace devm {
struct UpdateInfo {
    ::String version;
    ::String packgName;
    ::String deviceType;
    ::UInt8 mainSub;
    ::UInt8 versionStatus;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(version);
        fun(packgName);
        fun(deviceType);
        fun(mainSub);
        fun(versionStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(version);
        fun(packgName);
        fun(deviceType);
        fun(mainSub);
        fun(versionStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("version", version);
        fun("packgName", packgName);
        fun("deviceType", deviceType);
        fun("mainSub", mainSub);
        fun("versionStatus", versionStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("version", version);
        fun("packgName", packgName);
        fun("deviceType", deviceType);
        fun("mainSub", mainSub);
        fun("versionStatus", versionStatus);
    }

    bool operator==(const ::mdc::devm::UpdateInfo& t) const
    {
        return (version == t.version) && (packgName == t.packgName) && (deviceType == t.deviceType) && (mainSub == t.mainSub) && (versionStatus == t.versionStatus);
    }
};
} // namespace devm
} // namespace mdc


#endif // MDC_DEVM_IMPL_TYPE_UPDATEINFO_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_LOCATIONMONITOR_H
#define HOZON_LOCATION_IMPL_TYPE_LOCATIONMONITOR_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "hozon/location/impl_type_moduleresultvec.h"

namespace hozon {
namespace location {
struct LocationMonitor {
    ::hozon::common::CommonHeader header;
    ::UInt32 errorCode;
    ::UInt32 innerCode;
    ::UInt32 actionCode;
    ::hozon::location::ModuleResultVec moduleResults;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(errorCode);
        fun(innerCode);
        fun(actionCode);
        fun(moduleResults);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(errorCode);
        fun(innerCode);
        fun(actionCode);
        fun(moduleResults);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("errorCode", errorCode);
        fun("innerCode", innerCode);
        fun("actionCode", actionCode);
        fun("moduleResults", moduleResults);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("errorCode", errorCode);
        fun("innerCode", innerCode);
        fun("actionCode", actionCode);
        fun("moduleResults", moduleResults);
    }

    bool operator==(const ::hozon::location::LocationMonitor& t) const
    {
        return (header == t.header) && (errorCode == t.errorCode) && (innerCode == t.innerCode) && (actionCode == t.actionCode) && (moduleResults == t.moduleResults);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_LOCATIONMONITOR_H

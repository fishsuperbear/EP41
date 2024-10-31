/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_COMPONENTHISTORYRECORDTYPE_H
#define MDC_SWM_IMPL_TYPE_COMPONENTHISTORYRECORDTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64.h"
#include "impl_type_string.h"
#include "impl_type_uint8.h"
#include "impl_type_int32.h"

namespace mdc {
namespace swm {
struct ComponentHistoryRecordType {
    ::UInt64 startTime;
    ::UInt64 endTime;
    ::String devName;
    ::String pkgName;
    ::String oldVersion;
    ::String newVersion;
    ::UInt8 actionType;
    ::Int32 errorCode;
    ::Int32 status;
    ::String message;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(startTime);
        fun(endTime);
        fun(devName);
        fun(pkgName);
        fun(oldVersion);
        fun(newVersion);
        fun(actionType);
        fun(errorCode);
        fun(status);
        fun(message);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(startTime);
        fun(endTime);
        fun(devName);
        fun(pkgName);
        fun(oldVersion);
        fun(newVersion);
        fun(actionType);
        fun(errorCode);
        fun(status);
        fun(message);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("devName", devName);
        fun("pkgName", pkgName);
        fun("oldVersion", oldVersion);
        fun("newVersion", newVersion);
        fun("actionType", actionType);
        fun("errorCode", errorCode);
        fun("status", status);
        fun("message", message);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("devName", devName);
        fun("pkgName", pkgName);
        fun("oldVersion", oldVersion);
        fun("newVersion", newVersion);
        fun("actionType", actionType);
        fun("errorCode", errorCode);
        fun("status", status);
        fun("message", message);
    }

    bool operator==(const ::mdc::swm::ComponentHistoryRecordType& t) const
    {
        return (startTime == t.startTime) && (endTime == t.endTime) && (devName == t.devName) && (pkgName == t.pkgName) && (oldVersion == t.oldVersion) && (newVersion == t.newVersion) && (actionType == t.actionType) && (errorCode == t.errorCode) && (status == t.status) && (message == t.message);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_COMPONENTHISTORYRECORDTYPE_H

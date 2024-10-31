/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_SOFTWAREPACKAGEHISTORYRECORDTYPE_H
#define MDC_SWM_IMPL_TYPE_SOFTWAREPACKAGEHISTORYRECORDTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_int32.h"
#include "impl_type_uint64.h"
#include "impl_type_string.h"
#include "mdc/swm/impl_type_componenthistoryrecordvector.h"

namespace mdc {
namespace swm {
struct SoftwarePackageHistoryRecordType {
    ::Int32 id;
    ::UInt64 startTime;
    ::UInt64 endTime;
    ::String softwarePackageName;
    ::String version;
    ::Int32 errorCode;
    ::mdc::swm::ComponentHistoryRecordVector componentRecord;
    ::String message;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(id);
        fun(startTime);
        fun(endTime);
        fun(softwarePackageName);
        fun(version);
        fun(errorCode);
        fun(componentRecord);
        fun(message);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(id);
        fun(startTime);
        fun(endTime);
        fun(softwarePackageName);
        fun(version);
        fun(errorCode);
        fun(componentRecord);
        fun(message);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("id", id);
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("softwarePackageName", softwarePackageName);
        fun("version", version);
        fun("errorCode", errorCode);
        fun("componentRecord", componentRecord);
        fun("message", message);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("id", id);
        fun("startTime", startTime);
        fun("endTime", endTime);
        fun("softwarePackageName", softwarePackageName);
        fun("version", version);
        fun("errorCode", errorCode);
        fun("componentRecord", componentRecord);
        fun("message", message);
    }

    bool operator==(const ::mdc::swm::SoftwarePackageHistoryRecordType& t) const
    {
        return (id == t.id) && (startTime == t.startTime) && (endTime == t.endTime) && (softwarePackageName == t.softwarePackageName) && (version == t.version) && (errorCode == t.errorCode) && (componentRecord == t.componentRecord) && (message == t.message);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_SOFTWAREPACKAGEHISTORYRECORDTYPE_H

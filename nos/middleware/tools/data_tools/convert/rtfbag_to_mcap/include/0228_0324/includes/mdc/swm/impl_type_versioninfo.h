/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_VERSIONINFO_H
#define MDC_SWM_IMPL_TYPE_VERSIONINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_string.h"

namespace mdc {
namespace swm {
struct VersionInfo {
    ::UInt8 partition;
    ::UInt8 isRunning;
    ::String version;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(partition);
        fun(isRunning);
        fun(version);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(partition);
        fun(isRunning);
        fun(version);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("partition", partition);
        fun("isRunning", isRunning);
        fun("version", version);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("partition", partition);
        fun("isRunning", isRunning);
        fun("version", version);
    }

    bool operator==(const ::mdc::swm::VersionInfo& t) const
    {
        return (partition == t.partition) && (isRunning == t.isRunning) && (version == t.version);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_VERSIONINFO_H

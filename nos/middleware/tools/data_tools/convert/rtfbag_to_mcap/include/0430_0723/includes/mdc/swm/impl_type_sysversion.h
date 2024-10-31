/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_SYSVERSION_H
#define MDC_SWM_IMPL_TYPE_SYSVERSION_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"
#include "impl_type_int8.h"
#include "mdc/swm/impl_type_versionitemvector.h"

namespace mdc {
namespace swm {
struct SysVersion {
    ::String sysVersion;
    ::Int8 state;
    ::mdc::swm::VersionItemVector itemVersions;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(sysVersion);
        fun(state);
        fun(itemVersions);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(sysVersion);
        fun(state);
        fun(itemVersions);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("sysVersion", sysVersion);
        fun("state", state);
        fun("itemVersions", itemVersions);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("sysVersion", sysVersion);
        fun("state", state);
        fun("itemVersions", itemVersions);
    }

    bool operator==(const ::mdc::swm::SysVersion& t) const
    {
        return (sysVersion == t.sysVersion) && (state == t.state) && (itemVersions == t.itemVersions);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_SYSVERSION_H

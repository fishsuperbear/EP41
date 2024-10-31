/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_SWCLUSTERINFOTYPE_H
#define MDC_SWM_IMPL_TYPE_SWCLUSTERINFOTYPE_H
#include <cfloat>
#include <cmath>
#include "mdc/swm/impl_type_swnametype.h"
#include "mdc/swm/impl_type_strongrevisionlabelstring.h"
#include "mdc/swm/impl_type_swclusterstatetype.h"

namespace mdc {
namespace swm {
struct SwClusterInfoType {
    ::mdc::swm::SwNameType name;
    ::mdc::swm::StrongRevisionLabelString Version;
    ::mdc::swm::SwClusterStateType State;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(name);
        fun(Version);
        fun(State);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(name);
        fun(Version);
        fun(State);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("name", name);
        fun("Version", Version);
        fun("State", State);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("name", name);
        fun("Version", Version);
        fun("State", State);
    }

    bool operator==(const ::mdc::swm::SwClusterInfoType& t) const
    {
        return (name == t.name) && (Version == t.Version) && (State == t.State);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_SWCLUSTERINFOTYPE_H

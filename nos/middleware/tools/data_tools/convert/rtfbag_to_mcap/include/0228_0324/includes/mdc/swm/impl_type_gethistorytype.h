/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_IMPL_TYPE_GETHISTORYTYPE_H
#define MDC_SWM_IMPL_TYPE_GETHISTORYTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint64.h"
#include "mdc/swm/impl_type_swnametype.h"
#include "mdc/swm/impl_type_strongrevisionlabelstring.h"
#include "mdc/swm/impl_type_actiontype.h"
#include "mdc/swm/impl_type_resulttype.h"

namespace mdc {
namespace swm {
struct GetHistoryType {
    ::UInt64 Time;
    ::mdc::swm::SwNameType Name;
    ::mdc::swm::StrongRevisionLabelString Version;
    ::mdc::swm::ActionType Action;
    ::mdc::swm::ResultType Resolution;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Time);
        fun(Name);
        fun(Version);
        fun(Action);
        fun(Resolution);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Time);
        fun(Name);
        fun(Version);
        fun(Action);
        fun(Resolution);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Time", Time);
        fun("Name", Name);
        fun("Version", Version);
        fun("Action", Action);
        fun("Resolution", Resolution);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Time", Time);
        fun("Name", Name);
        fun("Version", Version);
        fun("Action", Action);
        fun("Resolution", Resolution);
    }

    bool operator==(const ::mdc::swm::GetHistoryType& t) const
    {
        return (Time == t.Time) && (Name == t.Name) && (Version == t.Version) && (Action == t.Action) && (Resolution == t.Resolution);
    }
};
} // namespace swm
} // namespace mdc


#endif // MDC_SWM_IMPL_TYPE_GETHISTORYTYPE_H

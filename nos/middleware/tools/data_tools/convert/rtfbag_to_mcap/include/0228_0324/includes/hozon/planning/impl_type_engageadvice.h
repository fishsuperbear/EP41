/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PLANNING_IMPL_TYPE_ENGAGEADVICE_H
#define HOZON_PLANNING_IMPL_TYPE_ENGAGEADVICE_H
#include <cfloat>
#include <cmath>
#include "hozon/planning/impl_type_advise.h"
#include "impl_type_string.h"

namespace hozon {
namespace planning {
struct EngageAdvice {
    ::hozon::planning::Advise advise;
    ::String reason;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(advise);
        fun(reason);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(advise);
        fun(reason);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("advise", advise);
        fun("reason", reason);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("advise", advise);
        fun("reason", reason);
    }

    bool operator==(const ::hozon::planning::EngageAdvice& t) const
    {
        return (advise == t.advise) && (reason == t.reason);
    }
};
} // namespace planning
} // namespace hozon


#endif // HOZON_PLANNING_IMPL_TYPE_ENGAGEADVICE_H

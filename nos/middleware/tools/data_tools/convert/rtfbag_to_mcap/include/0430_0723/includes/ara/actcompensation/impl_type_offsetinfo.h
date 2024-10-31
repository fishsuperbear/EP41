/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ACTCOMPENSATION_IMPL_TYPE_OFFSETINFO_H
#define ARA_ACTCOMPENSATION_IMPL_TYPE_OFFSETINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"

namespace ara {
namespace actcompensation {
struct OffsetInfo {
    ::Float axOffset;
    ::Float ayOffset;
    ::Float azOffset;
    ::Float yawRateOffset;
    ::Float pitchRateOffset;
    ::Float rollRateOffset;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(axOffset);
        fun(ayOffset);
        fun(azOffset);
        fun(yawRateOffset);
        fun(pitchRateOffset);
        fun(rollRateOffset);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(axOffset);
        fun(ayOffset);
        fun(azOffset);
        fun(yawRateOffset);
        fun(pitchRateOffset);
        fun(rollRateOffset);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("axOffset", axOffset);
        fun("ayOffset", ayOffset);
        fun("azOffset", azOffset);
        fun("yawRateOffset", yawRateOffset);
        fun("pitchRateOffset", pitchRateOffset);
        fun("rollRateOffset", rollRateOffset);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("axOffset", axOffset);
        fun("ayOffset", ayOffset);
        fun("azOffset", azOffset);
        fun("yawRateOffset", yawRateOffset);
        fun("pitchRateOffset", pitchRateOffset);
        fun("rollRateOffset", rollRateOffset);
    }

    bool operator==(const ::ara::actcompensation::OffsetInfo& t) const
    {
        return (fabs(static_cast<double>(axOffset - t.axOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(ayOffset - t.ayOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(azOffset - t.azOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(yawRateOffset - t.yawRateOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(pitchRateOffset - t.pitchRateOffset)) < DBL_EPSILON) && (fabs(static_cast<double>(rollRateOffset - t.rollRateOffset)) < DBL_EPSILON);
    }
};
} // namespace actcompensation
} // namespace ara


#endif // ARA_ACTCOMPENSATION_IMPL_TYPE_OFFSETINFO_H

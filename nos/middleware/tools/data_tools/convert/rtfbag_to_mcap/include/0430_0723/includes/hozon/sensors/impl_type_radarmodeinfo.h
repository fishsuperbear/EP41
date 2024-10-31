/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_RADARMODEINFO_H
#define HOZON_SENSORS_IMPL_TYPE_RADARMODEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"
#include "hozon/composite/impl_type_point.h"

namespace hozon {
namespace sensors {
struct RadarModeInfo {
    ::Double x;
    ::Double y;
    ::Double z;
    ::hozon::composite::Point rms;
    ::hozon::composite::Point quality;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(x);
        fun(y);
        fun(z);
        fun(rms);
        fun(quality);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
        fun(rms);
        fun(quality);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("rms", rms);
        fun("quality", quality);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
        fun("rms", rms);
        fun("quality", quality);
    }

    bool operator==(const ::hozon::sensors::RadarModeInfo& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON) && (rms == t.rms) && (quality == t.quality);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_RADARMODEINFO_H

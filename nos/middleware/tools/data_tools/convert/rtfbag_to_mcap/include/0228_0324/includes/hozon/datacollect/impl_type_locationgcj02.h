/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DATACOLLECT_IMPL_TYPE_LOCATIONGCJ02_H
#define HOZON_DATACOLLECT_IMPL_TYPE_LOCATIONGCJ02_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace datacollect {
struct LocationGCJ02 {
    double x;
    double y;
    double z;

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
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(x);
        fun(y);
        fun(z);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("x", x);
        fun("y", y);
        fun("z", z);
    }

    bool operator==(const ::hozon::datacollect::LocationGCJ02& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON);
    }
};
} // namespace datacollect
} // namespace hozon


#endif // HOZON_DATACOLLECT_IMPL_TYPE_LOCATIONGCJ02_H

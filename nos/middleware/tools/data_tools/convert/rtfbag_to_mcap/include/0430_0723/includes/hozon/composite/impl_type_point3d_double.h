/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_COMPOSITE_IMPL_TYPE_POINT3D_DOUBLE_H
#define HOZON_COMPOSITE_IMPL_TYPE_POINT3D_DOUBLE_H
#include <cfloat>
#include <cmath>

namespace hozon {
namespace composite {
struct Point3D_double {
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

    bool operator==(const ::hozon::composite::Point3D_double& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON);
    }
};
} // namespace composite
} // namespace hozon


#endif // HOZON_COMPOSITE_IMPL_TYPE_POINT3D_DOUBLE_H

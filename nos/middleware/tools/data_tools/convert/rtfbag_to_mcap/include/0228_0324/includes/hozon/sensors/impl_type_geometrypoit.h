/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_GEOMETRYPOIT_H
#define HOZON_SENSORS_IMPL_TYPE_GEOMETRYPOIT_H
#include <cfloat>
#include <cmath>
#include "impl_type_double.h"

namespace hozon {
namespace sensors {
struct GeometryPoit {
    ::Double x;
    ::Double y;
    ::Double z;

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

    bool operator==(const ::hozon::sensors::GeometryPoit& t) const
    {
        return (fabs(static_cast<double>(x - t.x)) < DBL_EPSILON) && (fabs(static_cast<double>(y - t.y)) < DBL_EPSILON) && (fabs(static_cast<double>(z - t.z)) < DBL_EPSILON);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_GEOMETRYPOIT_H

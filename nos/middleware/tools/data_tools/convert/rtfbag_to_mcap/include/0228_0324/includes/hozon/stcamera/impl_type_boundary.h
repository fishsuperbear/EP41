/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_BOUNDARY_H
#define HOZON_STCAMERA_IMPL_TYPE_BOUNDARY_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_double.h"

namespace hozon {
namespace stcamera {
struct Boundary {
    ::UInt8 available;
    ::UInt8 color;
    ::UInt8 quality;
    ::Double distance;
    ::UInt8 type;
    ::Double age;
    ::Double yaw_angle;
    ::Double curvature_far;
    ::Double curvature_rate_far;
    ::Double curvature_vertical;
    ::Double curvature_rate_vertical;
    ::Double curvature_near;
    ::Double curvature_rate_near;
    ::Double boundary_width;
    ::UInt8 valid_length_near;
    ::UInt8 valid_length_far;
    ::UInt8 valid_length_vertical;
    ::Double boundary_height;
    ::Double dash_length;
    ::Double void_length;
    ::Double dash_phase;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(available);
        fun(color);
        fun(quality);
        fun(distance);
        fun(type);
        fun(age);
        fun(yaw_angle);
        fun(curvature_far);
        fun(curvature_rate_far);
        fun(curvature_vertical);
        fun(curvature_rate_vertical);
        fun(curvature_near);
        fun(curvature_rate_near);
        fun(boundary_width);
        fun(valid_length_near);
        fun(valid_length_far);
        fun(valid_length_vertical);
        fun(boundary_height);
        fun(dash_length);
        fun(void_length);
        fun(dash_phase);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(available);
        fun(color);
        fun(quality);
        fun(distance);
        fun(type);
        fun(age);
        fun(yaw_angle);
        fun(curvature_far);
        fun(curvature_rate_far);
        fun(curvature_vertical);
        fun(curvature_rate_vertical);
        fun(curvature_near);
        fun(curvature_rate_near);
        fun(boundary_width);
        fun(valid_length_near);
        fun(valid_length_far);
        fun(valid_length_vertical);
        fun(boundary_height);
        fun(dash_length);
        fun(void_length);
        fun(dash_phase);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("available", available);
        fun("color", color);
        fun("quality", quality);
        fun("distance", distance);
        fun("type", type);
        fun("age", age);
        fun("yaw_angle", yaw_angle);
        fun("curvature_far", curvature_far);
        fun("curvature_rate_far", curvature_rate_far);
        fun("curvature_vertical", curvature_vertical);
        fun("curvature_rate_vertical", curvature_rate_vertical);
        fun("curvature_near", curvature_near);
        fun("curvature_rate_near", curvature_rate_near);
        fun("boundary_width", boundary_width);
        fun("valid_length_near", valid_length_near);
        fun("valid_length_far", valid_length_far);
        fun("valid_length_vertical", valid_length_vertical);
        fun("boundary_height", boundary_height);
        fun("dash_length", dash_length);
        fun("void_length", void_length);
        fun("dash_phase", dash_phase);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("available", available);
        fun("color", color);
        fun("quality", quality);
        fun("distance", distance);
        fun("type", type);
        fun("age", age);
        fun("yaw_angle", yaw_angle);
        fun("curvature_far", curvature_far);
        fun("curvature_rate_far", curvature_rate_far);
        fun("curvature_vertical", curvature_vertical);
        fun("curvature_rate_vertical", curvature_rate_vertical);
        fun("curvature_near", curvature_near);
        fun("curvature_rate_near", curvature_rate_near);
        fun("boundary_width", boundary_width);
        fun("valid_length_near", valid_length_near);
        fun("valid_length_far", valid_length_far);
        fun("valid_length_vertical", valid_length_vertical);
        fun("boundary_height", boundary_height);
        fun("dash_length", dash_length);
        fun("void_length", void_length);
        fun("dash_phase", dash_phase);
    }

    bool operator==(const ::hozon::stcamera::Boundary& t) const
    {
        return (available == t.available) && (color == t.color) && (quality == t.quality) && (fabs(static_cast<double>(distance - t.distance)) < DBL_EPSILON) && (type == t.type) && (fabs(static_cast<double>(age - t.age)) < DBL_EPSILON) && (fabs(static_cast<double>(yaw_angle - t.yaw_angle)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature_far - t.curvature_far)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature_rate_far - t.curvature_rate_far)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature_vertical - t.curvature_vertical)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature_rate_vertical - t.curvature_rate_vertical)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature_near - t.curvature_near)) < DBL_EPSILON) && (fabs(static_cast<double>(curvature_rate_near - t.curvature_rate_near)) < DBL_EPSILON) && (fabs(static_cast<double>(boundary_width - t.boundary_width)) < DBL_EPSILON) && (valid_length_near == t.valid_length_near) && (valid_length_far == t.valid_length_far) && (valid_length_vertical == t.valid_length_vertical) && (fabs(static_cast<double>(boundary_height - t.boundary_height)) < DBL_EPSILON) && (fabs(static_cast<double>(dash_length - t.dash_length)) < DBL_EPSILON) && (fabs(static_cast<double>(void_length - t.void_length)) < DBL_EPSILON) && (fabs(static_cast<double>(dash_phase - t.dash_phase)) < DBL_EPSILON);
    }
};
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_BOUNDARY_H

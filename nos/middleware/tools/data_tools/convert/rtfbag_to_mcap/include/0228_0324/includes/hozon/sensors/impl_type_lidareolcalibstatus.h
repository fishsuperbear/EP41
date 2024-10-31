/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_LIDAREOLCALIBSTATUS_H
#define HOZON_SENSORS_IMPL_TYPE_LIDAREOLCALIBSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace hozon {
namespace sensors {
struct LidarEolCalibStatus {
    ::UInt8 calib_status;
    ::Float rotationX;
    ::Float rotationY;
    float rotationZ;
    ::Float rotationW;
    ::Float translationX;
    ::Float translationY;
    ::Float translationZ;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(calib_status);
        fun(rotationX);
        fun(rotationY);
        fun(rotationZ);
        fun(rotationW);
        fun(translationX);
        fun(translationY);
        fun(translationZ);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(calib_status);
        fun(rotationX);
        fun(rotationY);
        fun(rotationZ);
        fun(rotationW);
        fun(translationX);
        fun(translationY);
        fun(translationZ);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("calib_status", calib_status);
        fun("rotationX", rotationX);
        fun("rotationY", rotationY);
        fun("rotationZ", rotationZ);
        fun("rotationW", rotationW);
        fun("translationX", translationX);
        fun("translationY", translationY);
        fun("translationZ", translationZ);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("calib_status", calib_status);
        fun("rotationX", rotationX);
        fun("rotationY", rotationY);
        fun("rotationZ", rotationZ);
        fun("rotationW", rotationW);
        fun("translationX", translationX);
        fun("translationY", translationY);
        fun("translationZ", translationZ);
    }

    bool operator==(const ::hozon::sensors::LidarEolCalibStatus& t) const
    {
        return (calib_status == t.calib_status) && (fabs(static_cast<double>(rotationX - t.rotationX)) < DBL_EPSILON) && (fabs(static_cast<double>(rotationY - t.rotationY)) < DBL_EPSILON) && (fabs(static_cast<double>(rotationZ - t.rotationZ)) < DBL_EPSILON) && (fabs(static_cast<double>(rotationW - t.rotationW)) < DBL_EPSILON) && (fabs(static_cast<double>(translationX - t.translationX)) < DBL_EPSILON) && (fabs(static_cast<double>(translationY - t.translationY)) < DBL_EPSILON) && (fabs(static_cast<double>(translationZ - t.translationZ)) < DBL_EPSILON);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_LIDAREOLCALIBSTATUS_H

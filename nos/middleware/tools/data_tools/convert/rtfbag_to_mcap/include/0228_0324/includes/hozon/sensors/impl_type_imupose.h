/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_IMUPOSE_H
#define HOZON_SENSORS_IMPL_TYPE_IMUPOSE_H
#include <cfloat>
#include <cmath>
#include "hozon/sensors/impl_type_geometrypoit.h"

namespace hozon {
namespace sensors {
struct ImuPose {
    ::hozon::sensors::GeometryPoit imuPosition;
    ::hozon::sensors::GeometryPoit eulerAngle;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(imuPosition);
        fun(eulerAngle);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(imuPosition);
        fun(eulerAngle);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("imuPosition", imuPosition);
        fun("eulerAngle", eulerAngle);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("imuPosition", imuPosition);
        fun("eulerAngle", eulerAngle);
    }

    bool operator==(const ::hozon::sensors::ImuPose& t) const
    {
        return (imuPosition == t.imuPosition) && (eulerAngle == t.eulerAngle);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_IMUPOSE_H

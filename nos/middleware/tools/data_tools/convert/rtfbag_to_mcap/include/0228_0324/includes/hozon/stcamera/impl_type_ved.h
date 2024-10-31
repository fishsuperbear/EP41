/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_IMPL_TYPE_VED_H
#define HOZON_STCAMERA_IMPL_TYPE_VED_H
#include <cfloat>
#include <cmath>
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace stcamera {
struct VED {
    ::Float ved_lat_acc;
    ::UInt8 ved_motion_state;
    ::Float ved_long_acc;
    ::Float ved_vehicle_speed;
    ::Float ved_yaw_rate;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ved_lat_acc);
        fun(ved_motion_state);
        fun(ved_long_acc);
        fun(ved_vehicle_speed);
        fun(ved_yaw_rate);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ved_lat_acc);
        fun(ved_motion_state);
        fun(ved_long_acc);
        fun(ved_vehicle_speed);
        fun(ved_yaw_rate);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ved_lat_acc", ved_lat_acc);
        fun("ved_motion_state", ved_motion_state);
        fun("ved_long_acc", ved_long_acc);
        fun("ved_vehicle_speed", ved_vehicle_speed);
        fun("ved_yaw_rate", ved_yaw_rate);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ved_lat_acc", ved_lat_acc);
        fun("ved_motion_state", ved_motion_state);
        fun("ved_long_acc", ved_long_acc);
        fun("ved_vehicle_speed", ved_vehicle_speed);
        fun("ved_yaw_rate", ved_yaw_rate);
    }

    bool operator==(const ::hozon::stcamera::VED& t) const
    {
        return (fabs(static_cast<double>(ved_lat_acc - t.ved_lat_acc)) < DBL_EPSILON) && (ved_motion_state == t.ved_motion_state) && (fabs(static_cast<double>(ved_long_acc - t.ved_long_acc)) < DBL_EPSILON) && (fabs(static_cast<double>(ved_vehicle_speed - t.ved_vehicle_speed)) < DBL_EPSILON) && (fabs(static_cast<double>(ved_yaw_rate - t.ved_yaw_rate)) < DBL_EPSILON);
    }
};
} // namespace stcamera
} // namespace hozon


#endif // HOZON_STCAMERA_IMPL_TYPE_VED_H

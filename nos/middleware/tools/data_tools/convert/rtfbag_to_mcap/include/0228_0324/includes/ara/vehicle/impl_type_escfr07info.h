/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_ESCFR07INFO_H
#define ARA_VEHICLE_IMPL_TYPE_ESCFR07INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace ara {
namespace vehicle {
struct ESCFr07Info {
    ::UInt8 lat_acc_sensor_value_v;
    ::Float lat_acc_sensor_value;
    ::UInt8 vehicle_dyn_yaw_rate_v;
    ::UInt8 long_acc_sensor_value_v;
    ::Float long_acc_sensor_value;
    ::Float vehicle_dyn_yaw_rate;
    ::UInt8 vehicle_spd_valid;
    ::Float vehicle_spd;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lat_acc_sensor_value_v);
        fun(lat_acc_sensor_value);
        fun(vehicle_dyn_yaw_rate_v);
        fun(long_acc_sensor_value_v);
        fun(long_acc_sensor_value);
        fun(vehicle_dyn_yaw_rate);
        fun(vehicle_spd_valid);
        fun(vehicle_spd);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lat_acc_sensor_value_v);
        fun(lat_acc_sensor_value);
        fun(vehicle_dyn_yaw_rate_v);
        fun(long_acc_sensor_value_v);
        fun(long_acc_sensor_value);
        fun(vehicle_dyn_yaw_rate);
        fun(vehicle_spd_valid);
        fun(vehicle_spd);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("lat_acc_sensor_value_v", lat_acc_sensor_value_v);
        fun("lat_acc_sensor_value", lat_acc_sensor_value);
        fun("vehicle_dyn_yaw_rate_v", vehicle_dyn_yaw_rate_v);
        fun("long_acc_sensor_value_v", long_acc_sensor_value_v);
        fun("long_acc_sensor_value", long_acc_sensor_value);
        fun("vehicle_dyn_yaw_rate", vehicle_dyn_yaw_rate);
        fun("vehicle_spd_valid", vehicle_spd_valid);
        fun("vehicle_spd", vehicle_spd);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("lat_acc_sensor_value_v", lat_acc_sensor_value_v);
        fun("lat_acc_sensor_value", lat_acc_sensor_value);
        fun("vehicle_dyn_yaw_rate_v", vehicle_dyn_yaw_rate_v);
        fun("long_acc_sensor_value_v", long_acc_sensor_value_v);
        fun("long_acc_sensor_value", long_acc_sensor_value);
        fun("vehicle_dyn_yaw_rate", vehicle_dyn_yaw_rate);
        fun("vehicle_spd_valid", vehicle_spd_valid);
        fun("vehicle_spd", vehicle_spd);
    }

    bool operator==(const ::ara::vehicle::ESCFr07Info& t) const
    {
        return (lat_acc_sensor_value_v == t.lat_acc_sensor_value_v) && (fabs(static_cast<double>(lat_acc_sensor_value - t.lat_acc_sensor_value)) < DBL_EPSILON) && (vehicle_dyn_yaw_rate_v == t.vehicle_dyn_yaw_rate_v) && (long_acc_sensor_value_v == t.long_acc_sensor_value_v) && (fabs(static_cast<double>(long_acc_sensor_value - t.long_acc_sensor_value)) < DBL_EPSILON) && (fabs(static_cast<double>(vehicle_dyn_yaw_rate - t.vehicle_dyn_yaw_rate)) < DBL_EPSILON) && (vehicle_spd_valid == t.vehicle_spd_valid) && (fabs(static_cast<double>(vehicle_spd - t.vehicle_spd)) < DBL_EPSILON);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_ESCFR07INFO_H

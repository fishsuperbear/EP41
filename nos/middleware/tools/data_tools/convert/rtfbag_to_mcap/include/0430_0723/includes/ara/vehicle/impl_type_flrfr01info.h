/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_FLRFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_FLRFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace ara {
namespace vehicle {
struct FLRFr01Info {
    float flr_acc_required_accel;
    ::uint8_t flr_adas_sndctrl_acc_onoff_fbk;
    ::uint8_t flr_sndctrl_acc_activt_driver_fbk;
    float flr_acc_comfort_band_upper;
    float flr_acc_comfort_band_lower;
    float flr_acc_jerk_upper_limit;
    ::uint8_t flr_acc_stop_req;
    float flr_acc_jerk_lower_limit;
    ::uint8_t flr_fr01_msg_counter;
    ::uint8_t flr_fr01_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(flr_acc_required_accel);
        fun(flr_adas_sndctrl_acc_onoff_fbk);
        fun(flr_sndctrl_acc_activt_driver_fbk);
        fun(flr_acc_comfort_band_upper);
        fun(flr_acc_comfort_band_lower);
        fun(flr_acc_jerk_upper_limit);
        fun(flr_acc_stop_req);
        fun(flr_acc_jerk_lower_limit);
        fun(flr_fr01_msg_counter);
        fun(flr_fr01_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(flr_acc_required_accel);
        fun(flr_adas_sndctrl_acc_onoff_fbk);
        fun(flr_sndctrl_acc_activt_driver_fbk);
        fun(flr_acc_comfort_band_upper);
        fun(flr_acc_comfort_band_lower);
        fun(flr_acc_jerk_upper_limit);
        fun(flr_acc_stop_req);
        fun(flr_acc_jerk_lower_limit);
        fun(flr_fr01_msg_counter);
        fun(flr_fr01_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("flr_acc_required_accel", flr_acc_required_accel);
        fun("flr_adas_sndctrl_acc_onoff_fbk", flr_adas_sndctrl_acc_onoff_fbk);
        fun("flr_sndctrl_acc_activt_driver_fbk", flr_sndctrl_acc_activt_driver_fbk);
        fun("flr_acc_comfort_band_upper", flr_acc_comfort_band_upper);
        fun("flr_acc_comfort_band_lower", flr_acc_comfort_band_lower);
        fun("flr_acc_jerk_upper_limit", flr_acc_jerk_upper_limit);
        fun("flr_acc_stop_req", flr_acc_stop_req);
        fun("flr_acc_jerk_lower_limit", flr_acc_jerk_lower_limit);
        fun("flr_fr01_msg_counter", flr_fr01_msg_counter);
        fun("flr_fr01_checksum", flr_fr01_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("flr_acc_required_accel", flr_acc_required_accel);
        fun("flr_adas_sndctrl_acc_onoff_fbk", flr_adas_sndctrl_acc_onoff_fbk);
        fun("flr_sndctrl_acc_activt_driver_fbk", flr_sndctrl_acc_activt_driver_fbk);
        fun("flr_acc_comfort_band_upper", flr_acc_comfort_band_upper);
        fun("flr_acc_comfort_band_lower", flr_acc_comfort_band_lower);
        fun("flr_acc_jerk_upper_limit", flr_acc_jerk_upper_limit);
        fun("flr_acc_stop_req", flr_acc_stop_req);
        fun("flr_acc_jerk_lower_limit", flr_acc_jerk_lower_limit);
        fun("flr_fr01_msg_counter", flr_fr01_msg_counter);
        fun("flr_fr01_checksum", flr_fr01_checksum);
    }

    bool operator==(const ::ara::vehicle::FLRFr01Info& t) const
    {
        return (fabs(static_cast<double>(flr_acc_required_accel - t.flr_acc_required_accel)) < DBL_EPSILON) && (flr_adas_sndctrl_acc_onoff_fbk == t.flr_adas_sndctrl_acc_onoff_fbk) && (flr_sndctrl_acc_activt_driver_fbk == t.flr_sndctrl_acc_activt_driver_fbk) && (fabs(static_cast<double>(flr_acc_comfort_band_upper - t.flr_acc_comfort_band_upper)) < DBL_EPSILON) && (fabs(static_cast<double>(flr_acc_comfort_band_lower - t.flr_acc_comfort_band_lower)) < DBL_EPSILON) && (fabs(static_cast<double>(flr_acc_jerk_upper_limit - t.flr_acc_jerk_upper_limit)) < DBL_EPSILON) && (flr_acc_stop_req == t.flr_acc_stop_req) && (fabs(static_cast<double>(flr_acc_jerk_lower_limit - t.flr_acc_jerk_lower_limit)) < DBL_EPSILON) && (flr_fr01_msg_counter == t.flr_fr01_msg_counter) && (flr_fr01_checksum == t.flr_fr01_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_FLRFR01INFO_H

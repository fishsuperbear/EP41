/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_SWSMFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_SWSMFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct SWSMFr01Info {
    ::UInt8 swsm_a_lka_sys_sw_sts;
    ::UInt8 swsm_a_set_minus_swsts;
    ::UInt8 swsm_a_res_plus_swsts;
    ::UInt8 swsm_a_cruise_distance_swsts;
    ::UInt8 swsm_a_cruise_cancel_swsts;
    ::UInt8 swsm_a_cruise_swsts;
    ::UInt8 region_a_sw_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(swsm_a_lka_sys_sw_sts);
        fun(swsm_a_set_minus_swsts);
        fun(swsm_a_res_plus_swsts);
        fun(swsm_a_cruise_distance_swsts);
        fun(swsm_a_cruise_cancel_swsts);
        fun(swsm_a_cruise_swsts);
        fun(region_a_sw_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(swsm_a_lka_sys_sw_sts);
        fun(swsm_a_set_minus_swsts);
        fun(swsm_a_res_plus_swsts);
        fun(swsm_a_cruise_distance_swsts);
        fun(swsm_a_cruise_cancel_swsts);
        fun(swsm_a_cruise_swsts);
        fun(region_a_sw_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("swsm_a_lka_sys_sw_sts", swsm_a_lka_sys_sw_sts);
        fun("swsm_a_set_minus_swsts", swsm_a_set_minus_swsts);
        fun("swsm_a_res_plus_swsts", swsm_a_res_plus_swsts);
        fun("swsm_a_cruise_distance_swsts", swsm_a_cruise_distance_swsts);
        fun("swsm_a_cruise_cancel_swsts", swsm_a_cruise_cancel_swsts);
        fun("swsm_a_cruise_swsts", swsm_a_cruise_swsts);
        fun("region_a_sw_error", region_a_sw_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("swsm_a_lka_sys_sw_sts", swsm_a_lka_sys_sw_sts);
        fun("swsm_a_set_minus_swsts", swsm_a_set_minus_swsts);
        fun("swsm_a_res_plus_swsts", swsm_a_res_plus_swsts);
        fun("swsm_a_cruise_distance_swsts", swsm_a_cruise_distance_swsts);
        fun("swsm_a_cruise_cancel_swsts", swsm_a_cruise_cancel_swsts);
        fun("swsm_a_cruise_swsts", swsm_a_cruise_swsts);
        fun("region_a_sw_error", region_a_sw_error);
    }

    bool operator==(const ::ara::vehicle::SWSMFr01Info& t) const
    {
        return (swsm_a_lka_sys_sw_sts == t.swsm_a_lka_sys_sw_sts) && (swsm_a_set_minus_swsts == t.swsm_a_set_minus_swsts) && (swsm_a_res_plus_swsts == t.swsm_a_res_plus_swsts) && (swsm_a_cruise_distance_swsts == t.swsm_a_cruise_distance_swsts) && (swsm_a_cruise_cancel_swsts == t.swsm_a_cruise_cancel_swsts) && (swsm_a_cruise_swsts == t.swsm_a_cruise_swsts) && (region_a_sw_error == t.region_a_sw_error);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_SWSMFR01INFO_H

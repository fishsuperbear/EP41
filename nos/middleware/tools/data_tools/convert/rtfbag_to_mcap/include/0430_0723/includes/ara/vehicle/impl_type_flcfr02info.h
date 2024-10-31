/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_FLCFR02INFO_H
#define ARA_VEHICLE_IMPL_TYPE_FLCFR02INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace ara {
namespace vehicle {
struct FLCFr02Info {
    ::uint8_t flc_ihbc_opt_usm;
    ::uint8_t flc_ihbc_sys_state;
    ::uint8_t flc_tsr_opt_usm;
    ::uint8_t flc_tsr_display_speed;
    ::uint8_t flc_ddw_warn_set_sta;
    ::uint8_t flc_ddw_opt_usm;
    ::uint8_t flc_tsr_sys_state;
    ::uint8_t flc_tsr_display_speed_valid;
    ::uint8_t flc_ddw_fail_info;
    ::uint8_t flc_ddw_level_display;
    ::uint8_t flc_fr02_msg_counter;
    ::uint8_t flc_fr02_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(flc_ihbc_opt_usm);
        fun(flc_ihbc_sys_state);
        fun(flc_tsr_opt_usm);
        fun(flc_tsr_display_speed);
        fun(flc_ddw_warn_set_sta);
        fun(flc_ddw_opt_usm);
        fun(flc_tsr_sys_state);
        fun(flc_tsr_display_speed_valid);
        fun(flc_ddw_fail_info);
        fun(flc_ddw_level_display);
        fun(flc_fr02_msg_counter);
        fun(flc_fr02_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(flc_ihbc_opt_usm);
        fun(flc_ihbc_sys_state);
        fun(flc_tsr_opt_usm);
        fun(flc_tsr_display_speed);
        fun(flc_ddw_warn_set_sta);
        fun(flc_ddw_opt_usm);
        fun(flc_tsr_sys_state);
        fun(flc_tsr_display_speed_valid);
        fun(flc_ddw_fail_info);
        fun(flc_ddw_level_display);
        fun(flc_fr02_msg_counter);
        fun(flc_fr02_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("flc_ihbc_opt_usm", flc_ihbc_opt_usm);
        fun("flc_ihbc_sys_state", flc_ihbc_sys_state);
        fun("flc_tsr_opt_usm", flc_tsr_opt_usm);
        fun("flc_tsr_display_speed", flc_tsr_display_speed);
        fun("flc_ddw_warn_set_sta", flc_ddw_warn_set_sta);
        fun("flc_ddw_opt_usm", flc_ddw_opt_usm);
        fun("flc_tsr_sys_state", flc_tsr_sys_state);
        fun("flc_tsr_display_speed_valid", flc_tsr_display_speed_valid);
        fun("flc_ddw_fail_info", flc_ddw_fail_info);
        fun("flc_ddw_level_display", flc_ddw_level_display);
        fun("flc_fr02_msg_counter", flc_fr02_msg_counter);
        fun("flc_fr02_checksum", flc_fr02_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("flc_ihbc_opt_usm", flc_ihbc_opt_usm);
        fun("flc_ihbc_sys_state", flc_ihbc_sys_state);
        fun("flc_tsr_opt_usm", flc_tsr_opt_usm);
        fun("flc_tsr_display_speed", flc_tsr_display_speed);
        fun("flc_ddw_warn_set_sta", flc_ddw_warn_set_sta);
        fun("flc_ddw_opt_usm", flc_ddw_opt_usm);
        fun("flc_tsr_sys_state", flc_tsr_sys_state);
        fun("flc_tsr_display_speed_valid", flc_tsr_display_speed_valid);
        fun("flc_ddw_fail_info", flc_ddw_fail_info);
        fun("flc_ddw_level_display", flc_ddw_level_display);
        fun("flc_fr02_msg_counter", flc_fr02_msg_counter);
        fun("flc_fr02_checksum", flc_fr02_checksum);
    }

    bool operator==(const ::ara::vehicle::FLCFr02Info& t) const
    {
        return (flc_ihbc_opt_usm == t.flc_ihbc_opt_usm) && (flc_ihbc_sys_state == t.flc_ihbc_sys_state) && (flc_tsr_opt_usm == t.flc_tsr_opt_usm) && (flc_tsr_display_speed == t.flc_tsr_display_speed) && (flc_ddw_warn_set_sta == t.flc_ddw_warn_set_sta) && (flc_ddw_opt_usm == t.flc_ddw_opt_usm) && (flc_tsr_sys_state == t.flc_tsr_sys_state) && (flc_tsr_display_speed_valid == t.flc_tsr_display_speed_valid) && (flc_ddw_fail_info == t.flc_ddw_fail_info) && (flc_ddw_level_display == t.flc_ddw_level_display) && (flc_fr02_msg_counter == t.flc_fr02_msg_counter) && (flc_fr02_checksum == t.flc_fr02_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_FLCFR02INFO_H

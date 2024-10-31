/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_ESCFR06INFO_H
#define ARA_VEHICLE_IMPL_TYPE_ESCFR06INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct ESCFr06Info {
    ::Boolean rl_whl_velocity_valid;
    ::Float rl_whl_velocity;
    ::Boolean rr_whl_velocity_valid;
    ::Float rr_whl_velocity;
    ::Boolean hdc_enabled;
    ::UInt8 rl_whl_dir;
    ::UInt8 rr_whl_dir;
    ::UInt8 esc_fr06_msgcounter;
    ::Boolean master_cyl_pressure_invalid;
    ::Float master_cyl_pressure;
    ::UInt8 esc_fr06_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(rl_whl_velocity_valid);
        fun(rl_whl_velocity);
        fun(rr_whl_velocity_valid);
        fun(rr_whl_velocity);
        fun(hdc_enabled);
        fun(rl_whl_dir);
        fun(rr_whl_dir);
        fun(esc_fr06_msgcounter);
        fun(master_cyl_pressure_invalid);
        fun(master_cyl_pressure);
        fun(esc_fr06_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(rl_whl_velocity_valid);
        fun(rl_whl_velocity);
        fun(rr_whl_velocity_valid);
        fun(rr_whl_velocity);
        fun(hdc_enabled);
        fun(rl_whl_dir);
        fun(rr_whl_dir);
        fun(esc_fr06_msgcounter);
        fun(master_cyl_pressure_invalid);
        fun(master_cyl_pressure);
        fun(esc_fr06_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("rl_whl_velocity_valid", rl_whl_velocity_valid);
        fun("rl_whl_velocity", rl_whl_velocity);
        fun("rr_whl_velocity_valid", rr_whl_velocity_valid);
        fun("rr_whl_velocity", rr_whl_velocity);
        fun("hdc_enabled", hdc_enabled);
        fun("rl_whl_dir", rl_whl_dir);
        fun("rr_whl_dir", rr_whl_dir);
        fun("esc_fr06_msgcounter", esc_fr06_msgcounter);
        fun("master_cyl_pressure_invalid", master_cyl_pressure_invalid);
        fun("master_cyl_pressure", master_cyl_pressure);
        fun("esc_fr06_checksum", esc_fr06_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("rl_whl_velocity_valid", rl_whl_velocity_valid);
        fun("rl_whl_velocity", rl_whl_velocity);
        fun("rr_whl_velocity_valid", rr_whl_velocity_valid);
        fun("rr_whl_velocity", rr_whl_velocity);
        fun("hdc_enabled", hdc_enabled);
        fun("rl_whl_dir", rl_whl_dir);
        fun("rr_whl_dir", rr_whl_dir);
        fun("esc_fr06_msgcounter", esc_fr06_msgcounter);
        fun("master_cyl_pressure_invalid", master_cyl_pressure_invalid);
        fun("master_cyl_pressure", master_cyl_pressure);
        fun("esc_fr06_checksum", esc_fr06_checksum);
    }

    bool operator==(const ::ara::vehicle::ESCFr06Info& t) const
    {
        return (rl_whl_velocity_valid == t.rl_whl_velocity_valid) && (fabs(static_cast<double>(rl_whl_velocity - t.rl_whl_velocity)) < DBL_EPSILON) && (rr_whl_velocity_valid == t.rr_whl_velocity_valid) && (fabs(static_cast<double>(rr_whl_velocity - t.rr_whl_velocity)) < DBL_EPSILON) && (hdc_enabled == t.hdc_enabled) && (rl_whl_dir == t.rl_whl_dir) && (rr_whl_dir == t.rr_whl_dir) && (esc_fr06_msgcounter == t.esc_fr06_msgcounter) && (master_cyl_pressure_invalid == t.master_cyl_pressure_invalid) && (fabs(static_cast<double>(master_cyl_pressure - t.master_cyl_pressure)) < DBL_EPSILON) && (esc_fr06_checksum == t.esc_fr06_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_ESCFR06INFO_H

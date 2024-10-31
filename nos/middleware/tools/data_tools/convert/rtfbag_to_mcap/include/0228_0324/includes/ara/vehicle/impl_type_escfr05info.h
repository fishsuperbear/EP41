/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_ESCFR05INFO_H
#define ARA_VEHICLE_IMPL_TYPE_ESCFR05INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_float.h"
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct ESCFr05Info {
    ::Boolean fl_whl_velocity_valid;
    ::Float fl_whl_velocity;
    ::Boolean fr_whl_velocity_valid;
    ::Float fr_whl_velocity;
    ::UInt8 fr_whl_dir;
    ::UInt8 fl_whl_dir;
    ::UInt8 esc_fr05_msg_counter;
    ::UInt8 esc_fr05_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(fl_whl_velocity_valid);
        fun(fl_whl_velocity);
        fun(fr_whl_velocity_valid);
        fun(fr_whl_velocity);
        fun(fr_whl_dir);
        fun(fl_whl_dir);
        fun(esc_fr05_msg_counter);
        fun(esc_fr05_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(fl_whl_velocity_valid);
        fun(fl_whl_velocity);
        fun(fr_whl_velocity_valid);
        fun(fr_whl_velocity);
        fun(fr_whl_dir);
        fun(fl_whl_dir);
        fun(esc_fr05_msg_counter);
        fun(esc_fr05_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("fl_whl_velocity_valid", fl_whl_velocity_valid);
        fun("fl_whl_velocity", fl_whl_velocity);
        fun("fr_whl_velocity_valid", fr_whl_velocity_valid);
        fun("fr_whl_velocity", fr_whl_velocity);
        fun("fr_whl_dir", fr_whl_dir);
        fun("fl_whl_dir", fl_whl_dir);
        fun("esc_fr05_msg_counter", esc_fr05_msg_counter);
        fun("esc_fr05_checksum", esc_fr05_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("fl_whl_velocity_valid", fl_whl_velocity_valid);
        fun("fl_whl_velocity", fl_whl_velocity);
        fun("fr_whl_velocity_valid", fr_whl_velocity_valid);
        fun("fr_whl_velocity", fr_whl_velocity);
        fun("fr_whl_dir", fr_whl_dir);
        fun("fl_whl_dir", fl_whl_dir);
        fun("esc_fr05_msg_counter", esc_fr05_msg_counter);
        fun("esc_fr05_checksum", esc_fr05_checksum);
    }

    bool operator==(const ::ara::vehicle::ESCFr05Info& t) const
    {
        return (fl_whl_velocity_valid == t.fl_whl_velocity_valid) && (fabs(static_cast<double>(fl_whl_velocity - t.fl_whl_velocity)) < DBL_EPSILON) && (fr_whl_velocity_valid == t.fr_whl_velocity_valid) && (fabs(static_cast<double>(fr_whl_velocity - t.fr_whl_velocity)) < DBL_EPSILON) && (fr_whl_dir == t.fr_whl_dir) && (fl_whl_dir == t.fl_whl_dir) && (esc_fr05_msg_counter == t.esc_fr05_msg_counter) && (esc_fr05_checksum == t.esc_fr05_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_ESCFR05INFO_H

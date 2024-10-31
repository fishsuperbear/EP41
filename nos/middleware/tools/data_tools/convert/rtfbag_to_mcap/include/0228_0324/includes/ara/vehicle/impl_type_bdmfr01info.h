/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_BDMFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_BDMFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct BDMFr01Info {
    ::UInt8 bdm_back_door_ajar;
    ::UInt8 bdm_driver_door_ajar;
    ::UInt8 bdm_pass_door_ajar;
    ::UInt8 bdm_rear_driver_door_ajar;
    ::UInt8 bdm_rear_pass_door_ajar;
    ::UInt8 bdm_low_beam_st;
    ::UInt8 bdm_left_turn_light_st;
    ::UInt8 bdm_right_turn_light_st;
    ::UInt8 bdm_high_beam_st;
    ::UInt8 bdm_hazard_lamp_st;
    ::UInt8 bdm_front_lamp_sw;
    ::UInt8 bdm_high_beam_sw;
    ::UInt8 bdm_turn_light_sw;
    ::UInt8 bdm_front_wiper_st;
    ::UInt8 bdm_hazard_sw_st;
    ::UInt8 bdm_wiper_park_position;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(bdm_back_door_ajar);
        fun(bdm_driver_door_ajar);
        fun(bdm_pass_door_ajar);
        fun(bdm_rear_driver_door_ajar);
        fun(bdm_rear_pass_door_ajar);
        fun(bdm_low_beam_st);
        fun(bdm_left_turn_light_st);
        fun(bdm_right_turn_light_st);
        fun(bdm_high_beam_st);
        fun(bdm_hazard_lamp_st);
        fun(bdm_front_lamp_sw);
        fun(bdm_high_beam_sw);
        fun(bdm_turn_light_sw);
        fun(bdm_front_wiper_st);
        fun(bdm_hazard_sw_st);
        fun(bdm_wiper_park_position);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(bdm_back_door_ajar);
        fun(bdm_driver_door_ajar);
        fun(bdm_pass_door_ajar);
        fun(bdm_rear_driver_door_ajar);
        fun(bdm_rear_pass_door_ajar);
        fun(bdm_low_beam_st);
        fun(bdm_left_turn_light_st);
        fun(bdm_right_turn_light_st);
        fun(bdm_high_beam_st);
        fun(bdm_hazard_lamp_st);
        fun(bdm_front_lamp_sw);
        fun(bdm_high_beam_sw);
        fun(bdm_turn_light_sw);
        fun(bdm_front_wiper_st);
        fun(bdm_hazard_sw_st);
        fun(bdm_wiper_park_position);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("bdm_back_door_ajar", bdm_back_door_ajar);
        fun("bdm_driver_door_ajar", bdm_driver_door_ajar);
        fun("bdm_pass_door_ajar", bdm_pass_door_ajar);
        fun("bdm_rear_driver_door_ajar", bdm_rear_driver_door_ajar);
        fun("bdm_rear_pass_door_ajar", bdm_rear_pass_door_ajar);
        fun("bdm_low_beam_st", bdm_low_beam_st);
        fun("bdm_left_turn_light_st", bdm_left_turn_light_st);
        fun("bdm_right_turn_light_st", bdm_right_turn_light_st);
        fun("bdm_high_beam_st", bdm_high_beam_st);
        fun("bdm_hazard_lamp_st", bdm_hazard_lamp_st);
        fun("bdm_front_lamp_sw", bdm_front_lamp_sw);
        fun("bdm_high_beam_sw", bdm_high_beam_sw);
        fun("bdm_turn_light_sw", bdm_turn_light_sw);
        fun("bdm_front_wiper_st", bdm_front_wiper_st);
        fun("bdm_hazard_sw_st", bdm_hazard_sw_st);
        fun("bdm_wiper_park_position", bdm_wiper_park_position);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("bdm_back_door_ajar", bdm_back_door_ajar);
        fun("bdm_driver_door_ajar", bdm_driver_door_ajar);
        fun("bdm_pass_door_ajar", bdm_pass_door_ajar);
        fun("bdm_rear_driver_door_ajar", bdm_rear_driver_door_ajar);
        fun("bdm_rear_pass_door_ajar", bdm_rear_pass_door_ajar);
        fun("bdm_low_beam_st", bdm_low_beam_st);
        fun("bdm_left_turn_light_st", bdm_left_turn_light_st);
        fun("bdm_right_turn_light_st", bdm_right_turn_light_st);
        fun("bdm_high_beam_st", bdm_high_beam_st);
        fun("bdm_hazard_lamp_st", bdm_hazard_lamp_st);
        fun("bdm_front_lamp_sw", bdm_front_lamp_sw);
        fun("bdm_high_beam_sw", bdm_high_beam_sw);
        fun("bdm_turn_light_sw", bdm_turn_light_sw);
        fun("bdm_front_wiper_st", bdm_front_wiper_st);
        fun("bdm_hazard_sw_st", bdm_hazard_sw_st);
        fun("bdm_wiper_park_position", bdm_wiper_park_position);
    }

    bool operator==(const ::ara::vehicle::BDMFr01Info& t) const
    {
        return (bdm_back_door_ajar == t.bdm_back_door_ajar) && (bdm_driver_door_ajar == t.bdm_driver_door_ajar) && (bdm_pass_door_ajar == t.bdm_pass_door_ajar) && (bdm_rear_driver_door_ajar == t.bdm_rear_driver_door_ajar) && (bdm_rear_pass_door_ajar == t.bdm_rear_pass_door_ajar) && (bdm_low_beam_st == t.bdm_low_beam_st) && (bdm_left_turn_light_st == t.bdm_left_turn_light_st) && (bdm_right_turn_light_st == t.bdm_right_turn_light_st) && (bdm_high_beam_st == t.bdm_high_beam_st) && (bdm_hazard_lamp_st == t.bdm_hazard_lamp_st) && (bdm_front_lamp_sw == t.bdm_front_lamp_sw) && (bdm_high_beam_sw == t.bdm_high_beam_sw) && (bdm_turn_light_sw == t.bdm_turn_light_sw) && (bdm_front_wiper_st == t.bdm_front_wiper_st) && (bdm_hazard_sw_st == t.bdm_hazard_sw_st) && (bdm_wiper_park_position == t.bdm_wiper_park_position);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_BDMFR01INFO_H

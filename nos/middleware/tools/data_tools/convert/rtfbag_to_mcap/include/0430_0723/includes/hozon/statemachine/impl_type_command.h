/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STATEMACHINE_IMPL_TYPE_COMMAND_H
#define HOZON_STATEMACHINE_IMPL_TYPE_COMMAND_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace statemachine {
struct Command {
    ::UInt8 enable_parking_slot_detection;
    ::UInt8 enable_object_detection;
    ::UInt8 enable_freespace_detection;
    ::UInt8 enable_uss;
    ::UInt8 enable_radar;
    ::UInt8 enable_lidar;
    ::UInt8 system_command;
    ::UInt8 emergencybrake_state;
    ::UInt8 system_reset;
    ::uint8_t reserved1;
    ::uint8_t reserved2;
    ::uint8_t reserved3;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(enable_parking_slot_detection);
        fun(enable_object_detection);
        fun(enable_freespace_detection);
        fun(enable_uss);
        fun(enable_radar);
        fun(enable_lidar);
        fun(system_command);
        fun(emergencybrake_state);
        fun(system_reset);
        fun(reserved1);
        fun(reserved2);
        fun(reserved3);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(enable_parking_slot_detection);
        fun(enable_object_detection);
        fun(enable_freespace_detection);
        fun(enable_uss);
        fun(enable_radar);
        fun(enable_lidar);
        fun(system_command);
        fun(emergencybrake_state);
        fun(system_reset);
        fun(reserved1);
        fun(reserved2);
        fun(reserved3);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("enable_parking_slot_detection", enable_parking_slot_detection);
        fun("enable_object_detection", enable_object_detection);
        fun("enable_freespace_detection", enable_freespace_detection);
        fun("enable_uss", enable_uss);
        fun("enable_radar", enable_radar);
        fun("enable_lidar", enable_lidar);
        fun("system_command", system_command);
        fun("emergencybrake_state", emergencybrake_state);
        fun("system_reset", system_reset);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("enable_parking_slot_detection", enable_parking_slot_detection);
        fun("enable_object_detection", enable_object_detection);
        fun("enable_freespace_detection", enable_freespace_detection);
        fun("enable_uss", enable_uss);
        fun("enable_radar", enable_radar);
        fun("enable_lidar", enable_lidar);
        fun("system_command", system_command);
        fun("emergencybrake_state", emergencybrake_state);
        fun("system_reset", system_reset);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
    }

    bool operator==(const ::hozon::statemachine::Command& t) const
    {
        return (enable_parking_slot_detection == t.enable_parking_slot_detection) && (enable_object_detection == t.enable_object_detection) && (enable_freespace_detection == t.enable_freespace_detection) && (enable_uss == t.enable_uss) && (enable_radar == t.enable_radar) && (enable_lidar == t.enable_lidar) && (system_command == t.system_command) && (emergencybrake_state == t.emergencybrake_state) && (system_reset == t.system_reset) && (reserved1 == t.reserved1) && (reserved2 == t.reserved2) && (reserved3 == t.reserved3);
    }
};
} // namespace statemachine
} // namespace hozon


#endif // HOZON_STATEMACHINE_IMPL_TYPE_COMMAND_H

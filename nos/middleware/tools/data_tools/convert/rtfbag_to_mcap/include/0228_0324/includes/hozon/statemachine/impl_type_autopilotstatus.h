/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STATEMACHINE_IMPL_TYPE_AUTOPILOTSTATUS_H
#define HOZON_STATEMACHINE_IMPL_TYPE_AUTOPILOTSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace statemachine {
struct AutopilotStatus {
    ::UInt8 processing_status;
    ::UInt8 camera_status;
    ::UInt8 uss_status;
    ::UInt8 radar_status;
    ::UInt8 lidar_status;
    ::UInt8 velocity_status;
    ::UInt8 perception_status;
    ::UInt8 planning_status;
    ::UInt8 controlling_status;
    ::UInt8 turn_light_status;
    ::UInt8 localization_status;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(processing_status);
        fun(camera_status);
        fun(uss_status);
        fun(radar_status);
        fun(lidar_status);
        fun(velocity_status);
        fun(perception_status);
        fun(planning_status);
        fun(controlling_status);
        fun(turn_light_status);
        fun(localization_status);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(processing_status);
        fun(camera_status);
        fun(uss_status);
        fun(radar_status);
        fun(lidar_status);
        fun(velocity_status);
        fun(perception_status);
        fun(planning_status);
        fun(controlling_status);
        fun(turn_light_status);
        fun(localization_status);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("processing_status", processing_status);
        fun("camera_status", camera_status);
        fun("uss_status", uss_status);
        fun("radar_status", radar_status);
        fun("lidar_status", lidar_status);
        fun("velocity_status", velocity_status);
        fun("perception_status", perception_status);
        fun("planning_status", planning_status);
        fun("controlling_status", controlling_status);
        fun("turn_light_status", turn_light_status);
        fun("localization_status", localization_status);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("processing_status", processing_status);
        fun("camera_status", camera_status);
        fun("uss_status", uss_status);
        fun("radar_status", radar_status);
        fun("lidar_status", lidar_status);
        fun("velocity_status", velocity_status);
        fun("perception_status", perception_status);
        fun("planning_status", planning_status);
        fun("controlling_status", controlling_status);
        fun("turn_light_status", turn_light_status);
        fun("localization_status", localization_status);
    }

    bool operator==(const ::hozon::statemachine::AutopilotStatus& t) const
    {
        return (processing_status == t.processing_status) && (camera_status == t.camera_status) && (uss_status == t.uss_status) && (radar_status == t.radar_status) && (lidar_status == t.lidar_status) && (velocity_status == t.velocity_status) && (perception_status == t.perception_status) && (planning_status == t.planning_status) && (controlling_status == t.controlling_status) && (turn_light_status == t.turn_light_status) && (localization_status == t.localization_status);
    }
};
} // namespace statemachine
} // namespace hozon


#endif // HOZON_STATEMACHINE_IMPL_TYPE_AUTOPILOTSTATUS_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:  VehicleStatus
 */

#ifndef VIZ_VEHICLE_STATUS_H
#define VIZ_VEHICLE_STATUS_H

#include <cstdint>
#include "viz_header.h"

namespace mdc {
namespace visual {
enum GearState {
    /* For use when gear is left unspecified */
    NONE = 0,
    /* parking: used for Automatic Transmission */
    PARK = 1,
    /* reverse: used for Automatic Transmission */
    REVERSE = 2,
    /* neutral: used for Automatic Transmission */
    NEUTRAL = 3,
    /* drive: used for Automatic Transmission */
    DRIVE = 4,
    LOW = 5,
    HIGH = 6,
    /* used for Manual Transmisssion GEAR */
    GEAR_1 = 7,
    /* used for Manual Transmisssion GEAR */
    GEAR_2 = 8,
    /* used for Manual Transmisssion GEAR */
    GEAR_3 = 9,
    /* used for Manual Transmisssion GEAR */
    GEAR_4 = 10,
    /* used for Manual Transmisssion GEAR */
    GEAR_5 = 11,
    /* used for Manual Transmisssion GEAR */
    GEAR_6 = 12,
    /* used for Manual Transmisssion GEAR */
    GEAR_7 = 13,
    /* used for Manual Transmisssion GEAR */
    GEAR_8 = 14,
    /* used for Manual Transmisssion GEAR */
    GEAR_9 = 15,
};

enum TurnLightState {
    /* the turn light status(off) */
    TURN_LIGHT_NONE = 0,
    /* the turn light status(left on) */
    TURN_LIGHT_LEFT = 1,
    /* the turn light status(right on) */
    TURN_LIGHT_RIGHT = 2,
    /* the turn light status(left && right on) */
    TURN_LIGHT_DOUBLE = 3,
};

enum HazardLightState {
    /* the hazard lights status(off) */
    HAZARD_LIGHTS_OFF = 0,
    /* the hazard lights status(on) */
    HAZARD_LIGHTS_ON = 1,
};

struct VehicleStatus {
    Header header;
    /* Should only be assigned values from the enumerated constants above */
    uint8_t gear;
    /* steering Wheel in angle, clockwise/right is positive, 0 degrees is verticle/straight */
    double steeringWheelAngle;
    /* steering Wheel in radians, clockwise/right is positive, 0 degrees is verticle/straight */
    double steeringWheelRadians;
    /* km/h.  If the gear is Reverse, this will be negative. */
    double vehicleSpeed;
    /* m/s².  If the gear is Reverse, this will be negative. */
    double vehicleAcceleration;
    uint8_t turnLightState;
    uint8_t hazardLightState;
    /* percentage depressed between 0 and 1 (0 means not depressed, 1 means fully depressed), or by %. */
    double throttle;
    /* percentage depressed between 0 and 1 (0 means not depressed, 1 means fully depressed), or by %. */
    double brake;
    VehicleStatus()
        : header(),
          gear(0U),
          steeringWheelAngle(0.0),
          steeringWheelRadians(0.0),
          vehicleSpeed(0.0),
          vehicleAcceleration(0.0),
          turnLightState(0U),
          hazardLightState(0U),
          throttle(0.0),
          brake(0.0)
    {}
    VehicleStatus(const Header& vHeader, const uint8_t& vGear, const double& vSteeringWheelAngle,
    const double& vSteeringWheelRadians, const double& vVehicleSpeed, const double& vVehicleAcceleration,
    const uint8_t& vTurnLightState, const uint8_t& vHazardLightState, const double& vThrottle, const double& vBrake)
        : header(vHeader),
          gear(vGear),
          steeringWheelAngle(vSteeringWheelAngle),
          steeringWheelRadians(vSteeringWheelRadians),
          vehicleSpeed(vVehicleSpeed),
          vehicleAcceleration(vVehicleAcceleration),
          turnLightState(vTurnLightState),
          hazardLightState(vHazardLightState),
          throttle(vThrottle),
          brake(vBrake)
    {}
};
}
}

#endif // VIZ_VEHICLE_STATUS_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_EGOTRAJECTORY_IMPL_TYPE_EGOTRAJECTORY_H
#define ARA_EGOTRAJECTORY_IMPL_TYPE_EGOTRAJECTORY_H
#include <cfloat>
#include <cmath>
#include "ara/egotrajectory/impl_type_header.h"
#include "impl_type_uint8.h"
#include "impl_type_posewithcovariance.h"
#include "impl_type_twistwithcovariance.h"
#include "impl_type_accelwithcovariance.h"
#include "impl_type_double.h"
#include "impl_type_trajectorypointvector.h"
#include "impl_type_waypointvector.h"
#include "impl_type_boolean.h"
#include "impl_type_uint32.h"
#include "impl_type_string.h"
#include "ara/egotrajectory/impl_type_estop.h"
#include "impl_type_uint8_t.h"

namespace ara {
namespace egotrajectory {
struct EgoTrajectory {
    ::ara::egotrajectory::Header header;
    ::UInt8 locationState;
    ::PoseWithCovariance pose;
    ::TwistWithCovariance velocity;
    ::AccelWithCovariance acceleration;
    ::Double trajectoryLength;
    ::Double trajectoryPeriod;
    ::TrajectoryPointVector trajectoryPoints;
    ::WayPointVector wayPoints;
    ::Boolean isReplanning;
    ::UInt32 gear;
    ::ara::egotrajectory::Header routingHeader;
    ::String selfLaneId;
    ::UInt32 trajectoryType;
    ::String targetLaneId;
    ::ara::egotrajectory::Estop estop;
    ::uint8_t turnLight;
    bool isHold;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locationState);
        fun(pose);
        fun(velocity);
        fun(acceleration);
        fun(trajectoryLength);
        fun(trajectoryPeriod);
        fun(trajectoryPoints);
        fun(wayPoints);
        fun(isReplanning);
        fun(gear);
        fun(routingHeader);
        fun(selfLaneId);
        fun(trajectoryType);
        fun(targetLaneId);
        fun(estop);
        fun(turnLight);
        fun(isHold);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locationState);
        fun(pose);
        fun(velocity);
        fun(acceleration);
        fun(trajectoryLength);
        fun(trajectoryPeriod);
        fun(trajectoryPoints);
        fun(wayPoints);
        fun(isReplanning);
        fun(gear);
        fun(routingHeader);
        fun(selfLaneId);
        fun(trajectoryType);
        fun(targetLaneId);
        fun(estop);
        fun(turnLight);
        fun(isHold);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locationState", locationState);
        fun("pose", pose);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("trajectoryLength", trajectoryLength);
        fun("trajectoryPeriod", trajectoryPeriod);
        fun("trajectoryPoints", trajectoryPoints);
        fun("wayPoints", wayPoints);
        fun("isReplanning", isReplanning);
        fun("gear", gear);
        fun("routingHeader", routingHeader);
        fun("selfLaneId", selfLaneId);
        fun("trajectoryType", trajectoryType);
        fun("targetLaneId", targetLaneId);
        fun("estop", estop);
        fun("turnLight", turnLight);
        fun("isHold", isHold);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locationState", locationState);
        fun("pose", pose);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("trajectoryLength", trajectoryLength);
        fun("trajectoryPeriod", trajectoryPeriod);
        fun("trajectoryPoints", trajectoryPoints);
        fun("wayPoints", wayPoints);
        fun("isReplanning", isReplanning);
        fun("gear", gear);
        fun("routingHeader", routingHeader);
        fun("selfLaneId", selfLaneId);
        fun("trajectoryType", trajectoryType);
        fun("targetLaneId", targetLaneId);
        fun("estop", estop);
        fun("turnLight", turnLight);
        fun("isHold", isHold);
    }

    bool operator==(const ::ara::egotrajectory::EgoTrajectory& t) const
    {
        return (header == t.header) && (locationState == t.locationState) && (pose == t.pose) && (velocity == t.velocity) && (acceleration == t.acceleration) && (fabs(static_cast<double>(trajectoryLength - t.trajectoryLength)) < DBL_EPSILON) && (fabs(static_cast<double>(trajectoryPeriod - t.trajectoryPeriod)) < DBL_EPSILON) && (trajectoryPoints == t.trajectoryPoints) && (wayPoints == t.wayPoints) && (isReplanning == t.isReplanning) && (gear == t.gear) && (routingHeader == t.routingHeader) && (selfLaneId == t.selfLaneId) && (trajectoryType == t.trajectoryType) && (targetLaneId == t.targetLaneId) && (estop == t.estop) && (turnLight == t.turnLight) && (isHold == t.isHold);
    }
};
} // namespace egotrajectory
} // namespace ara


#endif // ARA_EGOTRAJECTORY_IMPL_TYPE_EGOTRAJECTORY_H

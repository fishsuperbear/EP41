/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_PLANNING_IMPL_TYPE_EGOTRAJECTORYFRAME_H
#define HOZON_PLANNING_IMPL_TYPE_EGOTRAJECTORYFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "impl_type_double.h"
#include "hozon/planning/impl_type_trajectorypointvector.h"
#include "hozon/planning/impl_type_estop.h"
#include "impl_type_boolean.h"
#include "impl_type_uint8.h"
#include "hozon/planning/impl_type_engageadvice.h"
#include "hozon/planning/impl_type_vehicalsignal.h"
#include "impl_type_int32.h"
#include "impl_type_uint16.h"
#include "hozon/planning/impl_type_uint16array_150.h"

namespace hozon {
namespace planning {
struct EgoTrajectoryFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 locSeq;
    ::Double trajectoryLength;
    ::Double trajectoryPeriod;
    ::hozon::planning::TrajectoryPointVector trajectoryPoints;
    ::UInt32 validPoints;
    ::hozon::planning::Estop estop;
    ::Boolean isReplanning;
    ::UInt8 gear;
    ::hozon::common::CommonHeader routingHeader;
    ::UInt32 selfLaneId;
    ::UInt32 trajectoryType;
    ::UInt32 targetLaneId;
    ::hozon::planning::EngageAdvice engageState;
    ::hozon::planning::VehicalSignal signal;
    ::Int32 ehp_count;
    ::UInt32 drivingMode;
    ::UInt32 functionMode;
    ::UInt16 utmZoneID;
    double proj_heading_offset;
    ::hozon::planning::uint16Array_150 reserve;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(locSeq);
        fun(trajectoryLength);
        fun(trajectoryPeriod);
        fun(trajectoryPoints);
        fun(validPoints);
        fun(estop);
        fun(isReplanning);
        fun(gear);
        fun(routingHeader);
        fun(selfLaneId);
        fun(trajectoryType);
        fun(targetLaneId);
        fun(engageState);
        fun(signal);
        fun(ehp_count);
        fun(drivingMode);
        fun(functionMode);
        fun(utmZoneID);
        fun(proj_heading_offset);
        fun(reserve);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(trajectoryLength);
        fun(trajectoryPeriod);
        fun(trajectoryPoints);
        fun(validPoints);
        fun(estop);
        fun(isReplanning);
        fun(gear);
        fun(routingHeader);
        fun(selfLaneId);
        fun(trajectoryType);
        fun(targetLaneId);
        fun(engageState);
        fun(signal);
        fun(ehp_count);
        fun(drivingMode);
        fun(functionMode);
        fun(utmZoneID);
        fun(proj_heading_offset);
        fun(reserve);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("trajectoryLength", trajectoryLength);
        fun("trajectoryPeriod", trajectoryPeriod);
        fun("trajectoryPoints", trajectoryPoints);
        fun("validPoints", validPoints);
        fun("estop", estop);
        fun("isReplanning", isReplanning);
        fun("gear", gear);
        fun("routingHeader", routingHeader);
        fun("selfLaneId", selfLaneId);
        fun("trajectoryType", trajectoryType);
        fun("targetLaneId", targetLaneId);
        fun("engageState", engageState);
        fun("signal", signal);
        fun("ehp_count", ehp_count);
        fun("drivingMode", drivingMode);
        fun("functionMode", functionMode);
        fun("utmZoneID", utmZoneID);
        fun("proj_heading_offset", proj_heading_offset);
        fun("reserve", reserve);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("trajectoryLength", trajectoryLength);
        fun("trajectoryPeriod", trajectoryPeriod);
        fun("trajectoryPoints", trajectoryPoints);
        fun("validPoints", validPoints);
        fun("estop", estop);
        fun("isReplanning", isReplanning);
        fun("gear", gear);
        fun("routingHeader", routingHeader);
        fun("selfLaneId", selfLaneId);
        fun("trajectoryType", trajectoryType);
        fun("targetLaneId", targetLaneId);
        fun("engageState", engageState);
        fun("signal", signal);
        fun("ehp_count", ehp_count);
        fun("drivingMode", drivingMode);
        fun("functionMode", functionMode);
        fun("utmZoneID", utmZoneID);
        fun("proj_heading_offset", proj_heading_offset);
        fun("reserve", reserve);
    }

    bool operator==(const ::hozon::planning::EgoTrajectoryFrame& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (fabs(static_cast<double>(trajectoryLength - t.trajectoryLength)) < DBL_EPSILON) && (fabs(static_cast<double>(trajectoryPeriod - t.trajectoryPeriod)) < DBL_EPSILON) && (trajectoryPoints == t.trajectoryPoints) && (validPoints == t.validPoints) && (estop == t.estop) && (isReplanning == t.isReplanning) && (gear == t.gear) && (routingHeader == t.routingHeader) && (selfLaneId == t.selfLaneId) && (trajectoryType == t.trajectoryType) && (targetLaneId == t.targetLaneId) && (engageState == t.engageState) && (signal == t.signal) && (ehp_count == t.ehp_count) && (drivingMode == t.drivingMode) && (functionMode == t.functionMode) && (utmZoneID == t.utmZoneID) && (fabs(static_cast<double>(proj_heading_offset - t.proj_heading_offset)) < DBL_EPSILON) && (reserve == t.reserve);
    }
};
} // namespace planning
} // namespace hozon


#endif // HOZON_PLANNING_IMPL_TYPE_EGOTRAJECTORYFRAME_H

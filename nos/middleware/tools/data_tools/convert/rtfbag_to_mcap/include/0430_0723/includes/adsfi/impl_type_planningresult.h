/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_IMPL_TYPE_PLANNINGRESULT_H
#define ADSFI_IMPL_TYPE_PLANNINGRESULT_H
#include <cfloat>
#include <cmath>
#include "ara/common/impl_type_commonheader.h"
#include "impl_type_double.h"
#include "adsfi/impl_type_trajectorypointvector.h"
#include "adsfi/impl_type_waypointvector.h"
#include "impl_type_boolean.h"
#include "impl_type_uint32.h"
#include "impl_type_string.h"
#include "adsfi/impl_type_estop.h"
#include "impl_type_uint8.h"

namespace adsfi {
struct PlanningResult {
    ::ara::common::CommonHeader header;
    ::Double trajectoryLength;
    ::Double trajectoryPeriod;
    ::adsfi::TrajectoryPointVector trajectoryPoints;
    ::adsfi::WayPointVector wayPoints;
    ::Boolean isReplanning;
    ::UInt32 gear;
    ::String selfLaneId;
    ::UInt32 trajectoryType;
    ::String targetLaneId;
    ::adsfi::Estop estop;
    ::UInt8 turnLight;
    ::Boolean isHold;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(trajectoryLength);
        fun(trajectoryPeriod);
        fun(trajectoryPoints);
        fun(wayPoints);
        fun(isReplanning);
        fun(gear);
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
        fun(trajectoryLength);
        fun(trajectoryPeriod);
        fun(trajectoryPoints);
        fun(wayPoints);
        fun(isReplanning);
        fun(gear);
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
        fun("trajectoryLength", trajectoryLength);
        fun("trajectoryPeriod", trajectoryPeriod);
        fun("trajectoryPoints", trajectoryPoints);
        fun("wayPoints", wayPoints);
        fun("isReplanning", isReplanning);
        fun("gear", gear);
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
        fun("trajectoryLength", trajectoryLength);
        fun("trajectoryPeriod", trajectoryPeriod);
        fun("trajectoryPoints", trajectoryPoints);
        fun("wayPoints", wayPoints);
        fun("isReplanning", isReplanning);
        fun("gear", gear);
        fun("selfLaneId", selfLaneId);
        fun("trajectoryType", trajectoryType);
        fun("targetLaneId", targetLaneId);
        fun("estop", estop);
        fun("turnLight", turnLight);
        fun("isHold", isHold);
    }

    bool operator==(const ::adsfi::PlanningResult& t) const
    {
        return (header == t.header) && (fabs(static_cast<double>(trajectoryLength - t.trajectoryLength)) < DBL_EPSILON) && (fabs(static_cast<double>(trajectoryPeriod - t.trajectoryPeriod)) < DBL_EPSILON) && (trajectoryPoints == t.trajectoryPoints) && (wayPoints == t.wayPoints) && (isReplanning == t.isReplanning) && (gear == t.gear) && (selfLaneId == t.selfLaneId) && (trajectoryType == t.trajectoryType) && (targetLaneId == t.targetLaneId) && (estop == t.estop) && (turnLight == t.turnLight) && (isHold == t.isHold);
    }
};
} // namespace adsfi


#endif // ADSFI_IMPL_TYPE_PLANNINGRESULT_H

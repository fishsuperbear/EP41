/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_EGOTRAJECTORYFRAME_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_EGOTRAJECTORYFRAME_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_commonheadertraj_soc_mcu.h"
#include "impl_type_uint32.h"
#include "impl_type_float.h"
#include "impl_type_double.h"
#include "hozon/soc_mcu/impl_type_trajectorypointarray_soc_mcu.h"
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"
#include "hozon/soc_mcu/impl_type_uint16array_150.h"

namespace hozon {
namespace soc_mcu {
struct EgoTrajectoryFrame_soc_mcu {
    ::hozon::soc_mcu::CommonHeaderTraj_soc_mcu header;
    ::UInt32 locSeq;
    ::Float trajectoryLength;
    ::Float trajectoryPeriod;
    ::Float proj_heading_offset;
    ::Double trajectoryPoint_reference_x;
    ::Double trajectoryPoint_reference_y;
    ::hozon::soc_mcu::TrajectoryPointArray_soc_mcu trajectoryPoints;
    ::UInt8 trajectoryValidPointsSize;
    ::UInt8 estop;
    ::Boolean isReplanning;
    ::UInt8 gear;
    ::UInt8 trajectoryType;
    ::UInt8 driviningMode;
    ::UInt8 functionMode;
    ::UInt8 utmZoneID;
    ::hozon::soc_mcu::uint16Array_150 reserve;

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
        fun(proj_heading_offset);
        fun(trajectoryPoint_reference_x);
        fun(trajectoryPoint_reference_y);
        fun(trajectoryPoints);
        fun(trajectoryValidPointsSize);
        fun(estop);
        fun(isReplanning);
        fun(gear);
        fun(trajectoryType);
        fun(driviningMode);
        fun(functionMode);
        fun(utmZoneID);
        fun(reserve);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(locSeq);
        fun(trajectoryLength);
        fun(trajectoryPeriod);
        fun(proj_heading_offset);
        fun(trajectoryPoint_reference_x);
        fun(trajectoryPoint_reference_y);
        fun(trajectoryPoints);
        fun(trajectoryValidPointsSize);
        fun(estop);
        fun(isReplanning);
        fun(gear);
        fun(trajectoryType);
        fun(driviningMode);
        fun(functionMode);
        fun(utmZoneID);
        fun(reserve);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("trajectoryLength", trajectoryLength);
        fun("trajectoryPeriod", trajectoryPeriod);
        fun("proj_heading_offset", proj_heading_offset);
        fun("trajectoryPoint_reference_x", trajectoryPoint_reference_x);
        fun("trajectoryPoint_reference_y", trajectoryPoint_reference_y);
        fun("trajectoryPoints", trajectoryPoints);
        fun("trajectoryValidPointsSize", trajectoryValidPointsSize);
        fun("estop", estop);
        fun("isReplanning", isReplanning);
        fun("gear", gear);
        fun("trajectoryType", trajectoryType);
        fun("driviningMode", driviningMode);
        fun("functionMode", functionMode);
        fun("utmZoneID", utmZoneID);
        fun("reserve", reserve);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("locSeq", locSeq);
        fun("trajectoryLength", trajectoryLength);
        fun("trajectoryPeriod", trajectoryPeriod);
        fun("proj_heading_offset", proj_heading_offset);
        fun("trajectoryPoint_reference_x", trajectoryPoint_reference_x);
        fun("trajectoryPoint_reference_y", trajectoryPoint_reference_y);
        fun("trajectoryPoints", trajectoryPoints);
        fun("trajectoryValidPointsSize", trajectoryValidPointsSize);
        fun("estop", estop);
        fun("isReplanning", isReplanning);
        fun("gear", gear);
        fun("trajectoryType", trajectoryType);
        fun("driviningMode", driviningMode);
        fun("functionMode", functionMode);
        fun("utmZoneID", utmZoneID);
        fun("reserve", reserve);
    }

    bool operator==(const ::hozon::soc_mcu::EgoTrajectoryFrame_soc_mcu& t) const
    {
        return (header == t.header) && (locSeq == t.locSeq) && (fabs(static_cast<double>(trajectoryLength - t.trajectoryLength)) < DBL_EPSILON) && (fabs(static_cast<double>(trajectoryPeriod - t.trajectoryPeriod)) < DBL_EPSILON) && (fabs(static_cast<double>(proj_heading_offset - t.proj_heading_offset)) < DBL_EPSILON) && (fabs(static_cast<double>(trajectoryPoint_reference_x - t.trajectoryPoint_reference_x)) < DBL_EPSILON) && (fabs(static_cast<double>(trajectoryPoint_reference_y - t.trajectoryPoint_reference_y)) < DBL_EPSILON) && (trajectoryPoints == t.trajectoryPoints) && (trajectoryValidPointsSize == t.trajectoryValidPointsSize) && (estop == t.estop) && (isReplanning == t.isReplanning) && (gear == t.gear) && (trajectoryType == t.trajectoryType) && (driviningMode == t.driviningMode) && (functionMode == t.functionMode) && (utmZoneID == t.utmZoneID) && (reserve == t.reserve);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_EGOTRAJECTORYFRAME_SOC_MCU_H

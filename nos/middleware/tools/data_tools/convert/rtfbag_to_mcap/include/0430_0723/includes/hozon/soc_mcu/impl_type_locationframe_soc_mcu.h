/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_LOCATIONFRAME_SOC_MCU_H
#define HOZON_SOC_MCU_IMPL_TYPE_LOCATIONFRAME_SOC_MCU_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_commonheader_soc_mcu.h"
#include "impl_type_uint32.h"
#include "impl_type_double.h"
#include "hozon/soc_mcu/impl_type_posewithcovariance_soc_mcu.h"
#include "hozon/soc_mcu/impl_type_twistwithcovariance_soc_mcu.h"
#include "hozon/soc_mcu/impl_type_accelwithcovariance_soc_mcu.h"
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct LocationFrame_soc_mcu {
    ::hozon::soc_mcu::CommonHeader_soc_mcu header;
    ::UInt32 gpsWeek;
    ::Double gpsSec;
    ::hozon::soc_mcu::PoseWithCovariance_soc_mcu pose;
    ::hozon::soc_mcu::TwistWithCovariance_soc_mcu velocity;
    ::hozon::soc_mcu::AccelWithCovariance_soc_mcu acceleration;
    ::UInt8 coordinateType;
    ::UInt8 rtkStatus;
    ::UInt8 locationState;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(pose);
        fun(velocity);
        fun(acceleration);
        fun(coordinateType);
        fun(rtkStatus);
        fun(locationState);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(pose);
        fun(velocity);
        fun(acceleration);
        fun(coordinateType);
        fun(rtkStatus);
        fun(locationState);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("pose", pose);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("coordinateType", coordinateType);
        fun("rtkStatus", rtkStatus);
        fun("locationState", locationState);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("pose", pose);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("coordinateType", coordinateType);
        fun("rtkStatus", rtkStatus);
        fun("locationState", locationState);
    }

    bool operator==(const ::hozon::soc_mcu::LocationFrame_soc_mcu& t) const
    {
        return (header == t.header) && (gpsWeek == t.gpsWeek) && (fabs(static_cast<double>(gpsSec - t.gpsSec)) < DBL_EPSILON) && (pose == t.pose) && (velocity == t.velocity) && (acceleration == t.acceleration) && (coordinateType == t.coordinateType) && (rtkStatus == t.rtkStatus) && (locationState == t.locationState);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_LOCATIONFRAME_SOC_MCU_H

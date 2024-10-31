/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LOCATION_IMPL_TYPE_LOCATIONFRAME_H
#define HOZON_LOCATION_IMPL_TYPE_LOCATIONFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/common/impl_type_commonheader.h"
#include "impl_type_uint32.h"
#include "impl_type_double.h"
#include "impl_type_int32.h"
#include "impl_type_uint8.h"
#include "hozon/composite/impl_type_vector3.h"
#include "hozon/composite/impl_type_posewithcovariance.h"
#include "hozon/composite/impl_type_twistwithcovariance.h"
#include "hozon/composite/impl_type_accelwithcovariance.h"

namespace hozon {
namespace location {
struct LocationFrame {
    ::hozon::common::CommonHeader header;
    ::UInt32 gpsWeek;
    ::Double gpsSec;
    ::Int32 received_ehp_counter;
    ::UInt8 coordinateType;
    ::hozon::composite::Vector3 mountingError;
    ::hozon::composite::PoseWithCovariance pose;
    ::hozon::composite::TwistWithCovariance velocity;
    ::hozon::composite::AccelWithCovariance acceleration;
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
        fun(received_ehp_counter);
        fun(coordinateType);
        fun(mountingError);
        fun(pose);
        fun(velocity);
        fun(acceleration);
        fun(rtkStatus);
        fun(locationState);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(gpsWeek);
        fun(gpsSec);
        fun(received_ehp_counter);
        fun(coordinateType);
        fun(mountingError);
        fun(pose);
        fun(velocity);
        fun(acceleration);
        fun(rtkStatus);
        fun(locationState);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("received_ehp_counter", received_ehp_counter);
        fun("coordinateType", coordinateType);
        fun("mountingError", mountingError);
        fun("pose", pose);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("rtkStatus", rtkStatus);
        fun("locationState", locationState);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("gpsWeek", gpsWeek);
        fun("gpsSec", gpsSec);
        fun("received_ehp_counter", received_ehp_counter);
        fun("coordinateType", coordinateType);
        fun("mountingError", mountingError);
        fun("pose", pose);
        fun("velocity", velocity);
        fun("acceleration", acceleration);
        fun("rtkStatus", rtkStatus);
        fun("locationState", locationState);
    }

    bool operator==(const ::hozon::location::LocationFrame& t) const
    {
        return (header == t.header) && (gpsWeek == t.gpsWeek) && (fabs(static_cast<double>(gpsSec - t.gpsSec)) < DBL_EPSILON) && (received_ehp_counter == t.received_ehp_counter) && (coordinateType == t.coordinateType) && (mountingError == t.mountingError) && (pose == t.pose) && (velocity == t.velocity) && (acceleration == t.acceleration) && (rtkStatus == t.rtkStatus) && (locationState == t.locationState);
    }
};
} // namespace location
} // namespace hozon


#endif // HOZON_LOCATION_IMPL_TYPE_LOCATIONFRAME_H

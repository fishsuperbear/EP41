/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_GNSSVELINFOFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_GNSSVELINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "hozon/sensors/impl_type_geometrypoit.h"

namespace hozon {
namespace sensors {
struct GnssVelInfoFrame {
    ::UInt8 solStatus;
    ::Float horSpd;
    ::Float trkGnd;
    ::hozon::sensors::GeometryPoit velocity;
    ::hozon::sensors::GeometryPoit velocityStd;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(solStatus);
        fun(horSpd);
        fun(trkGnd);
        fun(velocity);
        fun(velocityStd);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(solStatus);
        fun(horSpd);
        fun(trkGnd);
        fun(velocity);
        fun(velocityStd);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("solStatus", solStatus);
        fun("horSpd", horSpd);
        fun("trkGnd", trkGnd);
        fun("velocity", velocity);
        fun("velocityStd", velocityStd);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("solStatus", solStatus);
        fun("horSpd", horSpd);
        fun("trkGnd", trkGnd);
        fun("velocity", velocity);
        fun("velocityStd", velocityStd);
    }

    bool operator==(const ::hozon::sensors::GnssVelInfoFrame& t) const
    {
        return (solStatus == t.solStatus) && (fabs(static_cast<double>(horSpd - t.horSpd)) < DBL_EPSILON) && (fabs(static_cast<double>(trkGnd - t.trkGnd)) < DBL_EPSILON) && (velocity == t.velocity) && (velocityStd == t.velocityStd);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_GNSSVELINFOFRAME_H

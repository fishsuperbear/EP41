/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_IMU_IMPL_TYPE_IMURAWINFO_H
#define MDC_IMU_IMPL_TYPE_IMURAWINFO_H
#include <cfloat>
#include <cmath>
#include "ara/gnss/impl_type_header.h"
#include "ara/ins/impl_type_time.h"
#include "ara/gnss/impl_type_geometrypoit.h"
#include "impl_type_uint32.h"
#include "impl_type_float.h"
#include "impl_type_uint64.h"
#include "impl_type_double.h"

namespace mdc {
namespace imu {
struct ImuRawInfo {
    ::ara::gnss::Header header;
    ::ara::ins::Time sensorTime;
    ::ara::ins::Time sensorSyncTime;
    ::ara::gnss::GeometryPoit angularVelocity;
    ::ara::gnss::GeometryPoit acceleration;
    ::UInt32 imuStatus;
    ::UInt32 imuCalStatus;
    ::Float temperature;
    ::UInt64 reversed1;
    ::UInt64 reversed2;
    ::Double reversed3;
    ::Double reversed4;
    ::Double reversed5;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(sensorTime);
        fun(sensorSyncTime);
        fun(angularVelocity);
        fun(acceleration);
        fun(imuStatus);
        fun(imuCalStatus);
        fun(temperature);
        fun(reversed1);
        fun(reversed2);
        fun(reversed3);
        fun(reversed4);
        fun(reversed5);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(sensorTime);
        fun(sensorSyncTime);
        fun(angularVelocity);
        fun(acceleration);
        fun(imuStatus);
        fun(imuCalStatus);
        fun(temperature);
        fun(reversed1);
        fun(reversed2);
        fun(reversed3);
        fun(reversed4);
        fun(reversed5);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("sensorTime", sensorTime);
        fun("sensorSyncTime", sensorSyncTime);
        fun("angularVelocity", angularVelocity);
        fun("acceleration", acceleration);
        fun("imuStatus", imuStatus);
        fun("imuCalStatus", imuCalStatus);
        fun("temperature", temperature);
        fun("reversed1", reversed1);
        fun("reversed2", reversed2);
        fun("reversed3", reversed3);
        fun("reversed4", reversed4);
        fun("reversed5", reversed5);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("sensorTime", sensorTime);
        fun("sensorSyncTime", sensorSyncTime);
        fun("angularVelocity", angularVelocity);
        fun("acceleration", acceleration);
        fun("imuStatus", imuStatus);
        fun("imuCalStatus", imuCalStatus);
        fun("temperature", temperature);
        fun("reversed1", reversed1);
        fun("reversed2", reversed2);
        fun("reversed3", reversed3);
        fun("reversed4", reversed4);
        fun("reversed5", reversed5);
    }

    bool operator==(const ::mdc::imu::ImuRawInfo& t) const
    {
        return (header == t.header) && (sensorTime == t.sensorTime) && (sensorSyncTime == t.sensorSyncTime) && (angularVelocity == t.angularVelocity) && (acceleration == t.acceleration) && (imuStatus == t.imuStatus) && (imuCalStatus == t.imuCalStatus) && (fabs(static_cast<double>(temperature - t.temperature)) < DBL_EPSILON) && (reversed1 == t.reversed1) && (reversed2 == t.reversed2) && (fabs(static_cast<double>(reversed3 - t.reversed3)) < DBL_EPSILON) && (fabs(static_cast<double>(reversed4 - t.reversed4)) < DBL_EPSILON) && (fabs(static_cast<double>(reversed5 - t.reversed5)) < DBL_EPSILON);
    }
};
} // namespace imu
} // namespace mdc


#endif // MDC_IMU_IMPL_TYPE_IMURAWINFO_H

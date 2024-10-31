/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_IMU_IMPL_TYPE_IMUINFO_H
#define ARA_IMU_IMPL_TYPE_IMUINFO_H
#include <cfloat>
#include <cmath>
#include "ara/gnss/impl_type_header.h"
#include "ara/gnss/impl_type_geometrypoit.h"
#include "impl_type_uint16.h"
#include "impl_type_float.h"

namespace ara {
namespace imu {
struct ImuInfo {
    ::ara::gnss::Header header;
    ::ara::gnss::GeometryPoit angularVelocity;
    ::ara::gnss::GeometryPoit acceleration;
    ::UInt16 imuStatus;
    ::Float temperature;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(angularVelocity);
        fun(acceleration);
        fun(imuStatus);
        fun(temperature);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(angularVelocity);
        fun(acceleration);
        fun(imuStatus);
        fun(temperature);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("angularVelocity", angularVelocity);
        fun("acceleration", acceleration);
        fun("imuStatus", imuStatus);
        fun("temperature", temperature);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("angularVelocity", angularVelocity);
        fun("acceleration", acceleration);
        fun("imuStatus", imuStatus);
        fun("temperature", temperature);
    }

    bool operator==(const ::ara::imu::ImuInfo& t) const
    {
        return (header == t.header) && (angularVelocity == t.angularVelocity) && (acceleration == t.acceleration) && (imuStatus == t.imuStatus) && (fabs(static_cast<double>(temperature - t.temperature)) < DBL_EPSILON);
    }
};
} // namespace imu
} // namespace ara


#endif // ARA_IMU_IMPL_TYPE_IMUINFO_H

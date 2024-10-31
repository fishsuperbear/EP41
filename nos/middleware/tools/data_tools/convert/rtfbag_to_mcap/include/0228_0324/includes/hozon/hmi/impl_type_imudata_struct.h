/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_IMUDATA_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_IMUDATA_STRUCT_H
#include <cfloat>
#include <cmath>
#include "hozon/hmi/impl_type_imupoint_struct.h"

namespace hozon {
namespace hmi {
struct IMUData_Struct {
    ::hozon::hmi::IMUPoint_Struct AngularVelocity;
    ::hozon::hmi::IMUPoint_Struct LinearAcceleration;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(AngularVelocity);
        fun(LinearAcceleration);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(AngularVelocity);
        fun(LinearAcceleration);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("AngularVelocity", AngularVelocity);
        fun("LinearAcceleration", LinearAcceleration);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("AngularVelocity", AngularVelocity);
        fun("LinearAcceleration", LinearAcceleration);
    }

    bool operator==(const ::hozon::hmi::IMUData_Struct& t) const
    {
        return (AngularVelocity == t.AngularVelocity) && (LinearAcceleration == t.LinearAcceleration);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_IMUDATA_STRUCT_H

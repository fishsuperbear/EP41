/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_CHASSIS_IMPL_TYPE_ACTUATORWORKSTATUS_H
#define ARA_CHASSIS_IMPL_TYPE_ACTUATORWORKSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace chassis {
struct ActuatorWorkStatus {
    ::UInt8 capability;
    ::UInt8 response;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(capability);
        fun(response);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(capability);
        fun(response);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("capability", capability);
        fun("response", response);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("capability", capability);
        fun("response", response);
    }

    bool operator==(const ::ara::chassis::ActuatorWorkStatus& t) const
    {
        return (capability == t.capability) && (response == t.response);
    }
};
} // namespace chassis
} // namespace ara


#endif // ARA_CHASSIS_IMPL_TYPE_ACTUATORWORKSTATUS_H

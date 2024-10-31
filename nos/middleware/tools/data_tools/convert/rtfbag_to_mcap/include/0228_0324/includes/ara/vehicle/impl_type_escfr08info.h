/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_ESCFR08INFO_H
#define ARA_VEHICLE_IMPL_TYPE_ESCFR08INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct ESCFr08Info {
    ::UInt8 epb_status;
    ::UInt8 epb_available;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(epb_status);
        fun(epb_available);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(epb_status);
        fun(epb_available);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("epb_status", epb_status);
        fun("epb_available", epb_available);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("epb_status", epb_status);
        fun("epb_available", epb_available);
    }

    bool operator==(const ::ara::vehicle::ESCFr08Info& t) const
    {
        return (epb_status == t.epb_status) && (epb_available == t.epb_available);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_ESCFR08INFO_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_VCUFR05_223INFO_H
#define ARA_VEHICLE_IMPL_TYPE_VCUFR05_223INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct VCUFr05_223Info {
    ::UInt8 vcu5_actgear;
    ::UInt8 vcu5_actgear_valid;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vcu5_actgear);
        fun(vcu5_actgear_valid);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vcu5_actgear);
        fun(vcu5_actgear_valid);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vcu5_actgear", vcu5_actgear);
        fun("vcu5_actgear_valid", vcu5_actgear_valid);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vcu5_actgear", vcu5_actgear);
        fun("vcu5_actgear_valid", vcu5_actgear_valid);
    }

    bool operator==(const ::ara::vehicle::VCUFr05_223Info& t) const
    {
        return (vcu5_actgear == t.vcu5_actgear) && (vcu5_actgear_valid == t.vcu5_actgear_valid);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_VCUFR05_223INFO_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_VCUFR0FINFO_H
#define ARA_VEHICLE_IMPL_TYPE_VCUFR0FINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct VCUFr0fInfo {
    ::UInt8 vcu_sys_ready;
    ::UInt8 vcu_acc_ready;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(vcu_sys_ready);
        fun(vcu_acc_ready);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(vcu_sys_ready);
        fun(vcu_acc_ready);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("vcu_sys_ready", vcu_sys_ready);
        fun("vcu_acc_ready", vcu_acc_ready);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("vcu_sys_ready", vcu_sys_ready);
        fun("vcu_acc_ready", vcu_acc_ready);
    }

    bool operator==(const ::ara::vehicle::VCUFr0fInfo& t) const
    {
        return (vcu_sys_ready == t.vcu_sys_ready) && (vcu_acc_ready == t.vcu_acc_ready);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_VCUFR0FINFO_H

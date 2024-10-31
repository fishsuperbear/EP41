/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_ACUFR01INFO_H
#define ARA_VEHICLE_IMPL_TYPE_ACUFR01INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace ara {
namespace vehicle {
struct ACUFr01Info {
    ::UInt8 acu_drv_seat_beltr_st;
    ::UInt8 acu_sts_alive_counter;
    ::UInt8 acu_sts_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(acu_drv_seat_beltr_st);
        fun(acu_sts_alive_counter);
        fun(acu_sts_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(acu_drv_seat_beltr_st);
        fun(acu_sts_alive_counter);
        fun(acu_sts_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("acu_drv_seat_beltr_st", acu_drv_seat_beltr_st);
        fun("acu_sts_alive_counter", acu_sts_alive_counter);
        fun("acu_sts_checksum", acu_sts_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("acu_drv_seat_beltr_st", acu_drv_seat_beltr_st);
        fun("acu_sts_alive_counter", acu_sts_alive_counter);
        fun("acu_sts_checksum", acu_sts_checksum);
    }

    bool operator==(const ::ara::vehicle::ACUFr01Info& t) const
    {
        return (acu_drv_seat_beltr_st == t.acu_drv_seat_beltr_st) && (acu_sts_alive_counter == t.acu_sts_alive_counter) && (acu_sts_checksum == t.acu_sts_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_ACUFR01INFO_H

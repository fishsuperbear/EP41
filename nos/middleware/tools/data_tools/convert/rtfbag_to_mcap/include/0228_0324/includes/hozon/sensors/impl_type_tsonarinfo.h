/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_TSONARINFO_H
#define HOZON_SENSORS_IMPL_TYPE_TSONARINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace sensors {
struct tSonarInfo {
    ::UInt8 Tx_sensor_ID;
    ::UInt8 status;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Tx_sensor_ID);
        fun(status);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Tx_sensor_ID);
        fun(status);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Tx_sensor_ID", Tx_sensor_ID);
        fun("status", status);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Tx_sensor_ID", Tx_sensor_ID);
        fun("status", status);
    }

    bool operator==(const ::hozon::sensors::tSonarInfo& t) const
    {
        return (Tx_sensor_ID == t.Tx_sensor_ID) && (status == t.status);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_TSONARINFO_H

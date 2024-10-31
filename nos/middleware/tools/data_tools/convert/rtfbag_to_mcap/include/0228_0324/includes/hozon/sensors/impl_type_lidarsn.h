/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_LIDARSN_H
#define HOZON_SENSORS_IMPL_TYPE_LIDARSN_H
#include <cfloat>
#include <cmath>
#include "impl_type_string.h"

namespace hozon {
namespace sensors {
struct LidarSN {
    ::String ecuSerialNumber;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ecuSerialNumber);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ecuSerialNumber);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ecuSerialNumber", ecuSerialNumber);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ecuSerialNumber", ecuSerialNumber);
    }

    bool operator==(const ::hozon::sensors::LidarSN& t) const
    {
        return (ecuSerialNumber == t.ecuSerialNumber);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_LIDARSN_H

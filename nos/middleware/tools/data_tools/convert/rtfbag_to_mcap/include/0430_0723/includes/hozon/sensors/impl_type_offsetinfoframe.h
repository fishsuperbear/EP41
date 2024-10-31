/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_OFFSETINFOFRAME_H
#define HOZON_SENSORS_IMPL_TYPE_OFFSETINFOFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/sensors/impl_type_geometrypoit.h"

namespace hozon {
namespace sensors {
struct offsetInfoFrame {
    ::hozon::sensors::GeometryPoit gyoBias;
    ::hozon::sensors::GeometryPoit gyoSF;
    ::hozon::sensors::GeometryPoit accBias;
    ::hozon::sensors::GeometryPoit accSF;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(gyoBias);
        fun(gyoSF);
        fun(accBias);
        fun(accSF);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(gyoBias);
        fun(gyoSF);
        fun(accBias);
        fun(accSF);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("gyoBias", gyoBias);
        fun("gyoSF", gyoSF);
        fun("accBias", accBias);
        fun("accSF", accSF);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("gyoBias", gyoBias);
        fun("gyoSF", gyoSF);
        fun("accBias", accBias);
        fun("accSF", accSF);
    }

    bool operator==(const ::hozon::sensors::offsetInfoFrame& t) const
    {
        return (gyoBias == t.gyoBias) && (gyoSF == t.gyoSF) && (accBias == t.accBias) && (accSF == t.accSF);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_OFFSETINFOFRAME_H

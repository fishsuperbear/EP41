/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_APA_INFO_T_H
#define HOZON_SENSORS_IMPL_TYPE_APA_INFO_T_H
#include <cfloat>
#include <cmath>
#include "hozon/sensors/impl_type_tpointinfo_arry_20.h"

namespace hozon {
namespace sensors {
struct APA_Info_T {
    ::hozon::sensors::tPointInfo_Arry_20 ObstaclePoint_left;
    ::hozon::sensors::tPointInfo_Arry_20 ObstaclePoint_right;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ObstaclePoint_left);
        fun(ObstaclePoint_right);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ObstaclePoint_left);
        fun(ObstaclePoint_right);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ObstaclePoint_left", ObstaclePoint_left);
        fun("ObstaclePoint_right", ObstaclePoint_right);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ObstaclePoint_left", ObstaclePoint_left);
        fun("ObstaclePoint_right", ObstaclePoint_right);
    }

    bool operator==(const ::hozon::sensors::APA_Info_T& t) const
    {
        return (ObstaclePoint_left == t.ObstaclePoint_left) && (ObstaclePoint_right == t.ObstaclePoint_right);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_APA_INFO_T_H

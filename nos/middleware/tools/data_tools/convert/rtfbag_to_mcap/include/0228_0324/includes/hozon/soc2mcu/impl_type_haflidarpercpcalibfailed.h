/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFLIDARPERCPCALIBFAILED_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFLIDARPERCPCALIBFAILED_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc2mcu {
struct HafLidarPercpCalibFailed {
    ::UInt8 Calib_l_lidar_error;
    ::UInt8 Calib_r_lidar_error;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Calib_l_lidar_error);
        fun(Calib_r_lidar_error);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Calib_l_lidar_error);
        fun(Calib_r_lidar_error);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Calib_l_lidar_error", Calib_l_lidar_error);
        fun("Calib_r_lidar_error", Calib_r_lidar_error);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Calib_l_lidar_error", Calib_l_lidar_error);
        fun("Calib_r_lidar_error", Calib_r_lidar_error);
    }

    bool operator==(const ::hozon::soc2mcu::HafLidarPercpCalibFailed& t) const
    {
        return (Calib_l_lidar_error == t.Calib_l_lidar_error) && (Calib_r_lidar_error == t.Calib_r_lidar_error);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFLIDARPERCPCALIBFAILED_H

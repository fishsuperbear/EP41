/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFLIDARPERCPCALIBFAILED_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFLIDARPERCPCALIBFAILED_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_HafLidarPercpCalibFailed {
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

    bool operator==(const ::hozon::soc_mcu::DtDebug_HafLidarPercpCalibFailed& t) const
    {
        return (Calib_l_lidar_error == t.Calib_l_lidar_error) && (Calib_r_lidar_error == t.Calib_r_lidar_error);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFLIDARPERCPCALIBFAILED_H

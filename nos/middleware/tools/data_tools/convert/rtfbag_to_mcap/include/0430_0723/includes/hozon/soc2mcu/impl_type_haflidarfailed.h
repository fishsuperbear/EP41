/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFLIDARFAILED_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFLIDARFAILED_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/soc2mcu/impl_type_haflidarpercpalgofailed.h"
#include "hozon/soc2mcu/impl_type_haflidarpercpdatafailed.h"
#include "hozon/soc2mcu/impl_type_haflidarpercpalgointrfailed.h"
#include "hozon/soc2mcu/impl_type_haflidarpercpcalibfailed.h"

namespace hozon {
namespace soc2mcu {
struct HafLidarFailed {
    ::UInt8 Ins_data_Failed;
    ::hozon::soc2mcu::HafLidarPercpAlgoFailed Lidar_Percp_Algo_Failed;
    ::hozon::soc2mcu::HafLidarPercpDataFailed Lidar_Percp_Data_Failed;
    ::hozon::soc2mcu::HafLidarPercpAlgoIntrFailed Lidar_Percp_Algo_Intr_Failed;
    ::hozon::soc2mcu::HafLidarPercpCalibFailed Lidar_Percp_Calib_Failed;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Ins_data_Failed);
        fun(Lidar_Percp_Algo_Failed);
        fun(Lidar_Percp_Data_Failed);
        fun(Lidar_Percp_Algo_Intr_Failed);
        fun(Lidar_Percp_Calib_Failed);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Ins_data_Failed);
        fun(Lidar_Percp_Algo_Failed);
        fun(Lidar_Percp_Data_Failed);
        fun(Lidar_Percp_Algo_Intr_Failed);
        fun(Lidar_Percp_Calib_Failed);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Ins_data_Failed", Ins_data_Failed);
        fun("Lidar_Percp_Algo_Failed", Lidar_Percp_Algo_Failed);
        fun("Lidar_Percp_Data_Failed", Lidar_Percp_Data_Failed);
        fun("Lidar_Percp_Algo_Intr_Failed", Lidar_Percp_Algo_Intr_Failed);
        fun("Lidar_Percp_Calib_Failed", Lidar_Percp_Calib_Failed);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Ins_data_Failed", Ins_data_Failed);
        fun("Lidar_Percp_Algo_Failed", Lidar_Percp_Algo_Failed);
        fun("Lidar_Percp_Data_Failed", Lidar_Percp_Data_Failed);
        fun("Lidar_Percp_Algo_Intr_Failed", Lidar_Percp_Algo_Intr_Failed);
        fun("Lidar_Percp_Calib_Failed", Lidar_Percp_Calib_Failed);
    }

    bool operator==(const ::hozon::soc2mcu::HafLidarFailed& t) const
    {
        return (Ins_data_Failed == t.Ins_data_Failed) && (Lidar_Percp_Algo_Failed == t.Lidar_Percp_Algo_Failed) && (Lidar_Percp_Data_Failed == t.Lidar_Percp_Data_Failed) && (Lidar_Percp_Algo_Intr_Failed == t.Lidar_Percp_Algo_Intr_Failed) && (Lidar_Percp_Calib_Failed == t.Lidar_Percp_Calib_Failed);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFLIDARFAILED_H

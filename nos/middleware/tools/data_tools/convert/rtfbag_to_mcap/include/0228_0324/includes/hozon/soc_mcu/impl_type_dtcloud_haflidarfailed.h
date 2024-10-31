/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARFAILED_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARFAILED_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_dtcloud_haflidarpercpalgofailed.h"
#include "hozon/soc_mcu/impl_type_dtcloud_haflidarpercpdatafailed.h"
#include "hozon/soc_mcu/impl_type_dtcloud_haflidarpercpalgointrfailed.h"
#include "hozon/soc_mcu/impl_type_dtcloud_haflidarpercpcalibfailed.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafLidarFailed {
    ::UInt8 Ins_data_Failed;
    ::hozon::soc_mcu::DtCloud_HafLidarPercpAlgoFailed Lidar_Percp_Algo_Failed;
    ::hozon::soc_mcu::DtCloud_HafLidarPercpDataFailed Lidar_Percp_Data_Failed;
    ::hozon::soc_mcu::DtCloud_HafLidarPercpAlgoIntrFailed Lidar_Percp_Algo_Intr_Failed;
    ::hozon::soc_mcu::DtCloud_HafLidarPercpCalibFailed Lidar_Percp_Calib_Failed;

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

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafLidarFailed& t) const
    {
        return (Ins_data_Failed == t.Ins_data_Failed) && (Lidar_Percp_Algo_Failed == t.Lidar_Percp_Algo_Failed) && (Lidar_Percp_Data_Failed == t.Lidar_Percp_Data_Failed) && (Lidar_Percp_Algo_Intr_Failed == t.Lidar_Percp_Algo_Intr_Failed) && (Lidar_Percp_Calib_Failed == t.Lidar_Percp_Calib_Failed);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFLIDARFAILED_H

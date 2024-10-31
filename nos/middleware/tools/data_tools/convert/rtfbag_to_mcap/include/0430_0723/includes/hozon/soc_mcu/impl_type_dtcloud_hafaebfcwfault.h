/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFAEBFCWFAULT_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFAEBFCWFAULT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_dtcloud_haflidarfailed.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafAebFcwFault {
    ::UInt8 FID_AEBSYS_Failed;
    ::UInt8 FID_AEB_Failed;
    ::hozon::soc_mcu::DtCloud_HafLidarFailed Lidar_Failed;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(FID_AEBSYS_Failed);
        fun(FID_AEB_Failed);
        fun(Lidar_Failed);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(FID_AEBSYS_Failed);
        fun(FID_AEB_Failed);
        fun(Lidar_Failed);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("FID_AEBSYS_Failed", FID_AEBSYS_Failed);
        fun("FID_AEB_Failed", FID_AEB_Failed);
        fun("Lidar_Failed", Lidar_Failed);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("FID_AEBSYS_Failed", FID_AEBSYS_Failed);
        fun("FID_AEB_Failed", FID_AEB_Failed);
        fun("Lidar_Failed", Lidar_Failed);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafAebFcwFault& t) const
    {
        return (FID_AEBSYS_Failed == t.FID_AEBSYS_Failed) && (FID_AEB_Failed == t.FID_AEB_Failed) && (Lidar_Failed == t.Lidar_Failed);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFAEBFCWFAULT_H

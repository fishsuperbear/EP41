/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFAEBFCWFAULTBUSTYPE_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFAEBFCWFAULTBUSTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/soc2mcu/impl_type_haflidarfailed.h"

namespace hozon {
namespace soc2mcu {
struct HafAebFcwFaultBusType {
    ::UInt8 FID_AEBSYS_Failed;
    ::UInt8 FID_AEB_Failed;
    ::hozon::soc2mcu::HafLidarFailed Lidar_Failed;

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

    bool operator==(const ::hozon::soc2mcu::HafAebFcwFaultBusType& t) const
    {
        return (FID_AEBSYS_Failed == t.FID_AEBSYS_Failed) && (FID_AEB_Failed == t.FID_AEB_Failed) && (Lidar_Failed == t.Lidar_Failed);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFAEBFCWFAULTBUSTYPE_H

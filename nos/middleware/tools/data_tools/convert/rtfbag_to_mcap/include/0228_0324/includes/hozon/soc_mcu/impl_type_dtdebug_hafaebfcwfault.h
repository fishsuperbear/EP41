/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFAEBFCWFAULT_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFAEBFCWFAULT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_dtdebug_haflidarfailed.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_HafAebFcwFault {
    ::UInt8 unsignedcharFID_AEBSYS_Failed;
    ::UInt8 unsignedcharFID_AEB_Failed;
    ::hozon::soc_mcu::DtDebug_HafLidarFailed Lidar_Failed;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(unsignedcharFID_AEBSYS_Failed);
        fun(unsignedcharFID_AEB_Failed);
        fun(Lidar_Failed);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(unsignedcharFID_AEBSYS_Failed);
        fun(unsignedcharFID_AEB_Failed);
        fun(Lidar_Failed);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("unsignedcharFID_AEBSYS_Failed", unsignedcharFID_AEBSYS_Failed);
        fun("unsignedcharFID_AEB_Failed", unsignedcharFID_AEB_Failed);
        fun("Lidar_Failed", Lidar_Failed);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("unsignedcharFID_AEBSYS_Failed", unsignedcharFID_AEBSYS_Failed);
        fun("unsignedcharFID_AEB_Failed", unsignedcharFID_AEB_Failed);
        fun("Lidar_Failed", Lidar_Failed);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_HafAebFcwFault& t) const
    {
        return (unsignedcharFID_AEBSYS_Failed == t.unsignedcharFID_AEBSYS_Failed) && (unsignedcharFID_AEB_Failed == t.unsignedcharFID_AEB_Failed) && (Lidar_Failed == t.Lidar_Failed);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_HAFAEBFCWFAULT_H

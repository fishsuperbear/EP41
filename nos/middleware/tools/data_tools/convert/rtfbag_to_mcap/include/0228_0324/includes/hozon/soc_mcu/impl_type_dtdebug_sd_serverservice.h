/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_SERVERSERVICE_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_SERVERSERVICE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_Sd_ServerService {
    ::UInt8 AutoAvailable;
    ::UInt16 HandleId;
    ::UInt16 ServiceId;
    ::UInt16 InstanceId;
    ::UInt16 LoadBalancingPriority;
    ::UInt16 LoadBalancingWeight;
    ::UInt8 MajorVersion;
    ::UInt32 MinorVersion;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(AutoAvailable);
        fun(HandleId);
        fun(ServiceId);
        fun(InstanceId);
        fun(LoadBalancingPriority);
        fun(LoadBalancingWeight);
        fun(MajorVersion);
        fun(MinorVersion);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(AutoAvailable);
        fun(HandleId);
        fun(ServiceId);
        fun(InstanceId);
        fun(LoadBalancingPriority);
        fun(LoadBalancingWeight);
        fun(MajorVersion);
        fun(MinorVersion);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("AutoAvailable", AutoAvailable);
        fun("HandleId", HandleId);
        fun("ServiceId", ServiceId);
        fun("InstanceId", InstanceId);
        fun("LoadBalancingPriority", LoadBalancingPriority);
        fun("LoadBalancingWeight", LoadBalancingWeight);
        fun("MajorVersion", MajorVersion);
        fun("MinorVersion", MinorVersion);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("AutoAvailable", AutoAvailable);
        fun("HandleId", HandleId);
        fun("ServiceId", ServiceId);
        fun("InstanceId", InstanceId);
        fun("LoadBalancingPriority", LoadBalancingPriority);
        fun("LoadBalancingWeight", LoadBalancingWeight);
        fun("MajorVersion", MajorVersion);
        fun("MinorVersion", MinorVersion);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_Sd_ServerService& t) const
    {
        return (AutoAvailable == t.AutoAvailable) && (HandleId == t.HandleId) && (ServiceId == t.ServiceId) && (InstanceId == t.InstanceId) && (LoadBalancingPriority == t.LoadBalancingPriority) && (LoadBalancingWeight == t.LoadBalancingWeight) && (MajorVersion == t.MajorVersion) && (MinorVersion == t.MinorVersion);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_SERVERSERVICE_H

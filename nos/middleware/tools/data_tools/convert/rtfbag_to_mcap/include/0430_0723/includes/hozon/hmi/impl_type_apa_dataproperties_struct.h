/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_APA_DATAPROPERTIES_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_APA_DATAPROPERTIES_STRUCT_H
#include <cfloat>
#include <cmath>
#include "hozon/hmi/impl_type_locfusioninfo_struct.h"
#include "hozon/hmi/impl_type_dynamicsrobject_array.h"
#include "hozon/hmi/impl_type_staticsrobject_array.h"
#include "hozon/hmi/impl_type_lanedataproperties_array.h"
#include "hozon/hmi/impl_type_imudata_struct.h"
#include "impl_type_uint64_t.h"

namespace hozon {
namespace hmi {
struct APA_Dataproperties_Struct {
    ::hozon::hmi::LocFusionInfo_Struct LocFusionInfo;
    ::hozon::hmi::DynamicSRObject_Array DynamicSRdata;
    ::hozon::hmi::StaticSRObject_Array StaticSRData;
    ::hozon::hmi::LaneDataProperties_Array LaneData;
    ::hozon::hmi::IMUData_Struct IMUData;
    ::uint64_t TimeStamp;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(LocFusionInfo);
        fun(DynamicSRdata);
        fun(StaticSRData);
        fun(LaneData);
        fun(IMUData);
        fun(TimeStamp);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(LocFusionInfo);
        fun(DynamicSRdata);
        fun(StaticSRData);
        fun(LaneData);
        fun(IMUData);
        fun(TimeStamp);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("LocFusionInfo", LocFusionInfo);
        fun("DynamicSRdata", DynamicSRdata);
        fun("StaticSRData", StaticSRData);
        fun("LaneData", LaneData);
        fun("IMUData", IMUData);
        fun("TimeStamp", TimeStamp);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("LocFusionInfo", LocFusionInfo);
        fun("DynamicSRdata", DynamicSRdata);
        fun("StaticSRData", StaticSRData);
        fun("LaneData", LaneData);
        fun("IMUData", IMUData);
        fun("TimeStamp", TimeStamp);
    }

    bool operator==(const ::hozon::hmi::APA_Dataproperties_Struct& t) const
    {
        return (LocFusionInfo == t.LocFusionInfo) && (DynamicSRdata == t.DynamicSRdata) && (StaticSRData == t.StaticSRData) && (LaneData == t.LaneData) && (IMUData == t.IMUData) && (TimeStamp == t.TimeStamp);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_APA_DATAPROPERTIES_STRUCT_H

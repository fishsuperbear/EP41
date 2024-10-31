/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_LOCFUSIONINFO_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_LOCFUSIONINFO_STRUCT_H
#include <cfloat>
#include <cmath>
#include "hozon/hmi/impl_type_locallfusionpos_struct.h"
#include "hozon/hmi/impl_type_loclanefusionresultexternal_struct.h"

namespace hozon {
namespace hmi {
struct LocFusionInfo_Struct {
    ::hozon::hmi::LocAllFusionPos_Struct AllFusionPosResult;
    ::hozon::hmi::LocLaneFusionResultExternal_Struct LaneFusionResult;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(AllFusionPosResult);
        fun(LaneFusionResult);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(AllFusionPosResult);
        fun(LaneFusionResult);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("AllFusionPosResult", AllFusionPosResult);
        fun("LaneFusionResult", LaneFusionResult);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("AllFusionPosResult", AllFusionPosResult);
        fun("LaneFusionResult", LaneFusionResult);
    }

    bool operator==(const ::hozon::hmi::LocFusionInfo_Struct& t) const
    {
        return (AllFusionPosResult == t.AllFusionPosResult) && (LaneFusionResult == t.LaneFusionResult);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_LOCFUSIONINFO_STRUCT_H

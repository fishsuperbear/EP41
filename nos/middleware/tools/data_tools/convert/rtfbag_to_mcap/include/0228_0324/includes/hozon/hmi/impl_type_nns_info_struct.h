/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_IMPL_TYPE_NNS_INFO_STRUCT_H
#define HOZON_HMI_IMPL_TYPE_NNS_INFO_STRUCT_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16_t.h"
#include "impl_type_uint32_t.h"
#include "hozon/hmi/impl_type_nns_path_array.h"

namespace hozon {
namespace hmi {
struct NNS_Info_Struct {
    bool IsPublicRoad;
    bool IsReplan;
    ::uint16_t padding_u16_1;
    float nextRouteDis;
    ::uint32_t nextManeuverld;
    ::uint32_t pathPointSize;
    ::hozon::hmi::NNS_Path_Array nnsPathArray;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(IsPublicRoad);
        fun(IsReplan);
        fun(padding_u16_1);
        fun(nextRouteDis);
        fun(nextManeuverld);
        fun(pathPointSize);
        fun(nnsPathArray);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(IsPublicRoad);
        fun(IsReplan);
        fun(padding_u16_1);
        fun(nextRouteDis);
        fun(nextManeuverld);
        fun(pathPointSize);
        fun(nnsPathArray);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("IsPublicRoad", IsPublicRoad);
        fun("IsReplan", IsReplan);
        fun("padding_u16_1", padding_u16_1);
        fun("nextRouteDis", nextRouteDis);
        fun("nextManeuverld", nextManeuverld);
        fun("pathPointSize", pathPointSize);
        fun("nnsPathArray", nnsPathArray);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("IsPublicRoad", IsPublicRoad);
        fun("IsReplan", IsReplan);
        fun("padding_u16_1", padding_u16_1);
        fun("nextRouteDis", nextRouteDis);
        fun("nextManeuverld", nextManeuverld);
        fun("pathPointSize", pathPointSize);
        fun("nnsPathArray", nnsPathArray);
    }

    bool operator==(const ::hozon::hmi::NNS_Info_Struct& t) const
    {
        return (IsPublicRoad == t.IsPublicRoad) && (IsReplan == t.IsReplan) && (padding_u16_1 == t.padding_u16_1) && (fabs(static_cast<double>(nextRouteDis - t.nextRouteDis)) < DBL_EPSILON) && (nextManeuverld == t.nextManeuverld) && (pathPointSize == t.pathPointSize) && (nnsPathArray == t.nnsPathArray);
    }
};
} // namespace hmi
} // namespace hozon


#endif // HOZON_HMI_IMPL_TYPE_NNS_INFO_STRUCT_H

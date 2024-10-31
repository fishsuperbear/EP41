/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_SWSWITCHINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_SWSWITCHINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct SWSwitchInfo {
    ::uint8_t SWSM_A_CruiseSpeed_Add;
    ::uint8_t SWSM_A_CruiseSpeed_Minus;
    ::uint8_t SWSM_A_CruiseDistance_Add;
    ::uint8_t SWSM_A_CruiseDistance_Minus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SWSM_A_CruiseSpeed_Add);
        fun(SWSM_A_CruiseSpeed_Minus);
        fun(SWSM_A_CruiseDistance_Add);
        fun(SWSM_A_CruiseDistance_Minus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SWSM_A_CruiseSpeed_Add);
        fun(SWSM_A_CruiseSpeed_Minus);
        fun(SWSM_A_CruiseDistance_Add);
        fun(SWSM_A_CruiseDistance_Minus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SWSM_A_CruiseSpeed_Add", SWSM_A_CruiseSpeed_Add);
        fun("SWSM_A_CruiseSpeed_Minus", SWSM_A_CruiseSpeed_Minus);
        fun("SWSM_A_CruiseDistance_Add", SWSM_A_CruiseDistance_Add);
        fun("SWSM_A_CruiseDistance_Minus", SWSM_A_CruiseDistance_Minus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SWSM_A_CruiseSpeed_Add", SWSM_A_CruiseSpeed_Add);
        fun("SWSM_A_CruiseSpeed_Minus", SWSM_A_CruiseSpeed_Minus);
        fun("SWSM_A_CruiseDistance_Add", SWSM_A_CruiseDistance_Add);
        fun("SWSM_A_CruiseDistance_Minus", SWSM_A_CruiseDistance_Minus);
    }

    bool operator==(const ::hozon::chassis::SWSwitchInfo& t) const
    {
        return (SWSM_A_CruiseSpeed_Add == t.SWSM_A_CruiseSpeed_Add) && (SWSM_A_CruiseSpeed_Minus == t.SWSM_A_CruiseSpeed_Minus) && (SWSM_A_CruiseDistance_Add == t.SWSM_A_CruiseDistance_Add) && (SWSM_A_CruiseDistance_Minus == t.SWSM_A_CruiseDistance_Minus);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_SWSWITCHINFO_H

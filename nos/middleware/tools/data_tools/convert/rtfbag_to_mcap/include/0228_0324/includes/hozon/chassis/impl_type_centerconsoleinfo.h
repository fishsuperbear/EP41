/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_CENTERCONSOLEINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_CENTERCONSOLEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct CenterConsoleInfo {
    ::uint8_t TSR_OnOffSet;
    ::uint8_t TSR_OverspeedOnoffSet;
    ::uint8_t IHBC_SysSwState;
    ::uint8_t FactoryReset;
    ::uint8_t ResetAllSetup;
    ::uint8_t TSR_LimitOverspeedSet;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(TSR_OnOffSet);
        fun(TSR_OverspeedOnoffSet);
        fun(IHBC_SysSwState);
        fun(FactoryReset);
        fun(ResetAllSetup);
        fun(TSR_LimitOverspeedSet);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(TSR_OnOffSet);
        fun(TSR_OverspeedOnoffSet);
        fun(IHBC_SysSwState);
        fun(FactoryReset);
        fun(ResetAllSetup);
        fun(TSR_LimitOverspeedSet);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("TSR_OnOffSet", TSR_OnOffSet);
        fun("TSR_OverspeedOnoffSet", TSR_OverspeedOnoffSet);
        fun("IHBC_SysSwState", IHBC_SysSwState);
        fun("FactoryReset", FactoryReset);
        fun("ResetAllSetup", ResetAllSetup);
        fun("TSR_LimitOverspeedSet", TSR_LimitOverspeedSet);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("TSR_OnOffSet", TSR_OnOffSet);
        fun("TSR_OverspeedOnoffSet", TSR_OverspeedOnoffSet);
        fun("IHBC_SysSwState", IHBC_SysSwState);
        fun("FactoryReset", FactoryReset);
        fun("ResetAllSetup", ResetAllSetup);
        fun("TSR_LimitOverspeedSet", TSR_LimitOverspeedSet);
    }

    bool operator==(const ::hozon::chassis::CenterConsoleInfo& t) const
    {
        return (TSR_OnOffSet == t.TSR_OnOffSet) && (TSR_OverspeedOnoffSet == t.TSR_OverspeedOnoffSet) && (IHBC_SysSwState == t.IHBC_SysSwState) && (FactoryReset == t.FactoryReset) && (ResetAllSetup == t.ResetAllSetup) && (TSR_LimitOverspeedSet == t.TSR_LimitOverspeedSet);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_CENTERCONSOLEINFO_H

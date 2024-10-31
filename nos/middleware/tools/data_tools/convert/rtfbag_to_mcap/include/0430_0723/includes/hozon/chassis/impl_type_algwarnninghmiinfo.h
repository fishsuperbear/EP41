/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGWARNNINGHMIINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGWARNNINGHMIINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct AlgWarnningHmiInfo {
    ::uint8_t ADCS8_VoiceMode;
    ::uint8_t RCTA_OnOffSet;
    ::uint8_t FCTA_OnOffSet;
    ::uint8_t DOW_OnOffSet;
    ::uint8_t RCW_OnOffSet;
    ::uint8_t LCA_OnOffSet;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS8_VoiceMode);
        fun(RCTA_OnOffSet);
        fun(FCTA_OnOffSet);
        fun(DOW_OnOffSet);
        fun(RCW_OnOffSet);
        fun(LCA_OnOffSet);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS8_VoiceMode);
        fun(RCTA_OnOffSet);
        fun(FCTA_OnOffSet);
        fun(DOW_OnOffSet);
        fun(RCW_OnOffSet);
        fun(LCA_OnOffSet);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS8_VoiceMode", ADCS8_VoiceMode);
        fun("RCTA_OnOffSet", RCTA_OnOffSet);
        fun("FCTA_OnOffSet", FCTA_OnOffSet);
        fun("DOW_OnOffSet", DOW_OnOffSet);
        fun("RCW_OnOffSet", RCW_OnOffSet);
        fun("LCA_OnOffSet", LCA_OnOffSet);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS8_VoiceMode", ADCS8_VoiceMode);
        fun("RCTA_OnOffSet", RCTA_OnOffSet);
        fun("FCTA_OnOffSet", FCTA_OnOffSet);
        fun("DOW_OnOffSet", DOW_OnOffSet);
        fun("RCW_OnOffSet", RCW_OnOffSet);
        fun("LCA_OnOffSet", LCA_OnOffSet);
    }

    bool operator==(const ::hozon::chassis::AlgWarnningHmiInfo& t) const
    {
        return (ADCS8_VoiceMode == t.ADCS8_VoiceMode) && (RCTA_OnOffSet == t.RCTA_OnOffSet) && (FCTA_OnOffSet == t.FCTA_OnOffSet) && (DOW_OnOffSet == t.DOW_OnOffSet) && (RCW_OnOffSet == t.RCW_OnOffSet) && (LCA_OnOffSet == t.LCA_OnOffSet);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGWARNNINGHMIINFO_H

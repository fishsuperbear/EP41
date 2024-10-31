/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_ALGMCUEGOMEMMSG_H
#define IMPL_TYPE_ALGMCUEGOMEMMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

struct AlgMcuEgoMemMsg {
    ::UInt8 CDCS11_VoiceMode;
    ::UInt8 RCTA_OnOffSet_mem;
    ::UInt8 FCTA_OnOffSet_mem;
    ::UInt8 DOW_OnOffSet_mem;
    ::UInt8 RCW_OnOffSet_mem;
    ::UInt8 LCA_OnOffSet_mem;
    ::UInt8 TSR_OnOffSet_mem;
    ::UInt8 TSR_OverspeedOnOffSet_mem;
    ::UInt8 IHBC_OnOffSet_mem;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(CDCS11_VoiceMode);
        fun(RCTA_OnOffSet_mem);
        fun(FCTA_OnOffSet_mem);
        fun(DOW_OnOffSet_mem);
        fun(RCW_OnOffSet_mem);
        fun(LCA_OnOffSet_mem);
        fun(TSR_OnOffSet_mem);
        fun(TSR_OverspeedOnOffSet_mem);
        fun(IHBC_OnOffSet_mem);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(CDCS11_VoiceMode);
        fun(RCTA_OnOffSet_mem);
        fun(FCTA_OnOffSet_mem);
        fun(DOW_OnOffSet_mem);
        fun(RCW_OnOffSet_mem);
        fun(LCA_OnOffSet_mem);
        fun(TSR_OnOffSet_mem);
        fun(TSR_OverspeedOnOffSet_mem);
        fun(IHBC_OnOffSet_mem);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("CDCS11_VoiceMode", CDCS11_VoiceMode);
        fun("RCTA_OnOffSet_mem", RCTA_OnOffSet_mem);
        fun("FCTA_OnOffSet_mem", FCTA_OnOffSet_mem);
        fun("DOW_OnOffSet_mem", DOW_OnOffSet_mem);
        fun("RCW_OnOffSet_mem", RCW_OnOffSet_mem);
        fun("LCA_OnOffSet_mem", LCA_OnOffSet_mem);
        fun("TSR_OnOffSet_mem", TSR_OnOffSet_mem);
        fun("TSR_OverspeedOnOffSet_mem", TSR_OverspeedOnOffSet_mem);
        fun("IHBC_OnOffSet_mem", IHBC_OnOffSet_mem);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("CDCS11_VoiceMode", CDCS11_VoiceMode);
        fun("RCTA_OnOffSet_mem", RCTA_OnOffSet_mem);
        fun("FCTA_OnOffSet_mem", FCTA_OnOffSet_mem);
        fun("DOW_OnOffSet_mem", DOW_OnOffSet_mem);
        fun("RCW_OnOffSet_mem", RCW_OnOffSet_mem);
        fun("LCA_OnOffSet_mem", LCA_OnOffSet_mem);
        fun("TSR_OnOffSet_mem", TSR_OnOffSet_mem);
        fun("TSR_OverspeedOnOffSet_mem", TSR_OverspeedOnOffSet_mem);
        fun("IHBC_OnOffSet_mem", IHBC_OnOffSet_mem);
    }

    bool operator==(const ::AlgMcuEgoMemMsg& t) const
    {
        return (CDCS11_VoiceMode == t.CDCS11_VoiceMode) && (RCTA_OnOffSet_mem == t.RCTA_OnOffSet_mem) && (FCTA_OnOffSet_mem == t.FCTA_OnOffSet_mem) && (DOW_OnOffSet_mem == t.DOW_OnOffSet_mem) && (RCW_OnOffSet_mem == t.RCW_OnOffSet_mem) && (LCA_OnOffSet_mem == t.LCA_OnOffSet_mem) && (TSR_OnOffSet_mem == t.TSR_OnOffSet_mem) && (TSR_OverspeedOnOffSet_mem == t.TSR_OverspeedOnOffSet_mem) && (IHBC_OnOffSet_mem == t.IHBC_OnOffSet_mem);
    }
};


#endif // IMPL_TYPE_ALGMCUEGOMEMMSG_H

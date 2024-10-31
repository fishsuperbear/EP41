/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_ALGEGOWARNINGINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_ALGEGOWARNINGINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct AlgEgoWarningInfo {
    ::UInt8 LCARightWarnSt;
    ::uint8_t LCALeftWarnSt;
    ::UInt8 LCAFaultStatus;
    ::uint8_t LCAState;
    ::UInt8 DOWState;
    ::UInt8 DOWWarnAudioplay;
    ::UInt8 DOWLeftWarnSt;
    ::UInt8 DOWRightWarnSt;
    ::UInt8 DOWFaultStatus;
    ::UInt8 RCTAState;
    ::UInt8 RCTAWarnAudioplay;
    ::UInt8 RCTALeftWarnSt;
    ::UInt8 RCTARightWarnSt;
    ::UInt8 RCTAFaultStatus;
    ::UInt8 FCTAState;
    ::UInt8 FCTAWarnAudioplay;
    ::UInt8 FCTAObjType;
    ::UInt8 FCTALeftWarnSt;
    ::UInt8 FCTARightWarnSt;
    ::UInt8 FCTAFaultStatus;
    ::UInt8 RCWState;
    ::UInt8 RCWWarnAudioplay;
    ::UInt8 RCWWarnSt;
    ::UInt8 RCWFaultStatus;
    ::UInt8 Voice_Mode;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(LCARightWarnSt);
        fun(LCALeftWarnSt);
        fun(LCAFaultStatus);
        fun(LCAState);
        fun(DOWState);
        fun(DOWWarnAudioplay);
        fun(DOWLeftWarnSt);
        fun(DOWRightWarnSt);
        fun(DOWFaultStatus);
        fun(RCTAState);
        fun(RCTAWarnAudioplay);
        fun(RCTALeftWarnSt);
        fun(RCTARightWarnSt);
        fun(RCTAFaultStatus);
        fun(FCTAState);
        fun(FCTAWarnAudioplay);
        fun(FCTAObjType);
        fun(FCTALeftWarnSt);
        fun(FCTARightWarnSt);
        fun(FCTAFaultStatus);
        fun(RCWState);
        fun(RCWWarnAudioplay);
        fun(RCWWarnSt);
        fun(RCWFaultStatus);
        fun(Voice_Mode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(LCARightWarnSt);
        fun(LCALeftWarnSt);
        fun(LCAFaultStatus);
        fun(LCAState);
        fun(DOWState);
        fun(DOWWarnAudioplay);
        fun(DOWLeftWarnSt);
        fun(DOWRightWarnSt);
        fun(DOWFaultStatus);
        fun(RCTAState);
        fun(RCTAWarnAudioplay);
        fun(RCTALeftWarnSt);
        fun(RCTARightWarnSt);
        fun(RCTAFaultStatus);
        fun(FCTAState);
        fun(FCTAWarnAudioplay);
        fun(FCTAObjType);
        fun(FCTALeftWarnSt);
        fun(FCTARightWarnSt);
        fun(FCTAFaultStatus);
        fun(RCWState);
        fun(RCWWarnAudioplay);
        fun(RCWWarnSt);
        fun(RCWFaultStatus);
        fun(Voice_Mode);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("LCARightWarnSt", LCARightWarnSt);
        fun("LCALeftWarnSt", LCALeftWarnSt);
        fun("LCAFaultStatus", LCAFaultStatus);
        fun("LCAState", LCAState);
        fun("DOWState", DOWState);
        fun("DOWWarnAudioplay", DOWWarnAudioplay);
        fun("DOWLeftWarnSt", DOWLeftWarnSt);
        fun("DOWRightWarnSt", DOWRightWarnSt);
        fun("DOWFaultStatus", DOWFaultStatus);
        fun("RCTAState", RCTAState);
        fun("RCTAWarnAudioplay", RCTAWarnAudioplay);
        fun("RCTALeftWarnSt", RCTALeftWarnSt);
        fun("RCTARightWarnSt", RCTARightWarnSt);
        fun("RCTAFaultStatus", RCTAFaultStatus);
        fun("FCTAState", FCTAState);
        fun("FCTAWarnAudioplay", FCTAWarnAudioplay);
        fun("FCTAObjType", FCTAObjType);
        fun("FCTALeftWarnSt", FCTALeftWarnSt);
        fun("FCTARightWarnSt", FCTARightWarnSt);
        fun("FCTAFaultStatus", FCTAFaultStatus);
        fun("RCWState", RCWState);
        fun("RCWWarnAudioplay", RCWWarnAudioplay);
        fun("RCWWarnSt", RCWWarnSt);
        fun("RCWFaultStatus", RCWFaultStatus);
        fun("Voice_Mode", Voice_Mode);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("LCARightWarnSt", LCARightWarnSt);
        fun("LCALeftWarnSt", LCALeftWarnSt);
        fun("LCAFaultStatus", LCAFaultStatus);
        fun("LCAState", LCAState);
        fun("DOWState", DOWState);
        fun("DOWWarnAudioplay", DOWWarnAudioplay);
        fun("DOWLeftWarnSt", DOWLeftWarnSt);
        fun("DOWRightWarnSt", DOWRightWarnSt);
        fun("DOWFaultStatus", DOWFaultStatus);
        fun("RCTAState", RCTAState);
        fun("RCTAWarnAudioplay", RCTAWarnAudioplay);
        fun("RCTALeftWarnSt", RCTALeftWarnSt);
        fun("RCTARightWarnSt", RCTARightWarnSt);
        fun("RCTAFaultStatus", RCTAFaultStatus);
        fun("FCTAState", FCTAState);
        fun("FCTAWarnAudioplay", FCTAWarnAudioplay);
        fun("FCTAObjType", FCTAObjType);
        fun("FCTALeftWarnSt", FCTALeftWarnSt);
        fun("FCTARightWarnSt", FCTARightWarnSt);
        fun("FCTAFaultStatus", FCTAFaultStatus);
        fun("RCWState", RCWState);
        fun("RCWWarnAudioplay", RCWWarnAudioplay);
        fun("RCWWarnSt", RCWWarnSt);
        fun("RCWFaultStatus", RCWFaultStatus);
        fun("Voice_Mode", Voice_Mode);
    }

    bool operator==(const ::hozon::chassis::AlgEgoWarningInfo& t) const
    {
        return (LCARightWarnSt == t.LCARightWarnSt) && (LCALeftWarnSt == t.LCALeftWarnSt) && (LCAFaultStatus == t.LCAFaultStatus) && (LCAState == t.LCAState) && (DOWState == t.DOWState) && (DOWWarnAudioplay == t.DOWWarnAudioplay) && (DOWLeftWarnSt == t.DOWLeftWarnSt) && (DOWRightWarnSt == t.DOWRightWarnSt) && (DOWFaultStatus == t.DOWFaultStatus) && (RCTAState == t.RCTAState) && (RCTAWarnAudioplay == t.RCTAWarnAudioplay) && (RCTALeftWarnSt == t.RCTALeftWarnSt) && (RCTARightWarnSt == t.RCTARightWarnSt) && (RCTAFaultStatus == t.RCTAFaultStatus) && (FCTAState == t.FCTAState) && (FCTAWarnAudioplay == t.FCTAWarnAudioplay) && (FCTAObjType == t.FCTAObjType) && (FCTALeftWarnSt == t.FCTALeftWarnSt) && (FCTARightWarnSt == t.FCTARightWarnSt) && (FCTAFaultStatus == t.FCTAFaultStatus) && (RCWState == t.RCWState) && (RCWWarnAudioplay == t.RCWWarnAudioplay) && (RCWWarnSt == t.RCWWarnSt) && (RCWFaultStatus == t.RCWFaultStatus) && (Voice_Mode == t.Voice_Mode);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_ALGEGOWARNINGINFO_H

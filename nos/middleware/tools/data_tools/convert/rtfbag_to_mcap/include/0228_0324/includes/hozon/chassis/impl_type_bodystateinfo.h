/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CHASSIS_IMPL_TYPE_BODYSTATEINFO_H
#define HOZON_CHASSIS_IMPL_TYPE_BODYSTATEINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"
#include "impl_type_uint8_t.h"

namespace hozon {
namespace chassis {
struct BodyStateInfo {
    ::UInt8 BCM_FLDrOpn;
    ::UInt8 BCM_FRDrOpn;
    ::UInt8 BCM_RLDrOpn;
    ::UInt8 BCM_RRDrOpn;
    ::UInt8 BCM_TGOpn;
    ::Boolean BCM_HodOpen;
    ::UInt8 BCM_DrvSeatbeltBucklesta;
    ::UInt8 BCM_FrontWiperSt;
    ::UInt8 BCM_HighBeamSt;
    ::UInt8 CS1_HighBeamReqSt ;
    ::UInt8 BCM_LowBeamSt;
    ::UInt8 HazardLampSt;
    ::Boolean BCM_FrontFogLampSt;
    ::Boolean BCM_RearFogLampSt;
    ::Boolean BCM_LeftTurnLightSt;
    ::Boolean BCM_RightTurnLightSt;
    ::UInt8 BCM_TurnLightSW;
    ::uint8_t BCM_FrontWiperWorkSts;
    ::uint8_t BCM_FrontLampSt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(BCM_FLDrOpn);
        fun(BCM_FRDrOpn);
        fun(BCM_RLDrOpn);
        fun(BCM_RRDrOpn);
        fun(BCM_TGOpn);
        fun(BCM_HodOpen);
        fun(BCM_DrvSeatbeltBucklesta);
        fun(BCM_FrontWiperSt);
        fun(BCM_HighBeamSt);
        fun(CS1_HighBeamReqSt );
        fun(BCM_LowBeamSt);
        fun(HazardLampSt);
        fun(BCM_FrontFogLampSt);
        fun(BCM_RearFogLampSt);
        fun(BCM_LeftTurnLightSt);
        fun(BCM_RightTurnLightSt);
        fun(BCM_TurnLightSW);
        fun(BCM_FrontWiperWorkSts);
        fun(BCM_FrontLampSt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(BCM_FLDrOpn);
        fun(BCM_FRDrOpn);
        fun(BCM_RLDrOpn);
        fun(BCM_RRDrOpn);
        fun(BCM_TGOpn);
        fun(BCM_HodOpen);
        fun(BCM_DrvSeatbeltBucklesta);
        fun(BCM_FrontWiperSt);
        fun(BCM_HighBeamSt);
        fun(CS1_HighBeamReqSt );
        fun(BCM_LowBeamSt);
        fun(HazardLampSt);
        fun(BCM_FrontFogLampSt);
        fun(BCM_RearFogLampSt);
        fun(BCM_LeftTurnLightSt);
        fun(BCM_RightTurnLightSt);
        fun(BCM_TurnLightSW);
        fun(BCM_FrontWiperWorkSts);
        fun(BCM_FrontLampSt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("BCM_FLDrOpn", BCM_FLDrOpn);
        fun("BCM_FRDrOpn", BCM_FRDrOpn);
        fun("BCM_RLDrOpn", BCM_RLDrOpn);
        fun("BCM_RRDrOpn", BCM_RRDrOpn);
        fun("BCM_TGOpn", BCM_TGOpn);
        fun("BCM_HodOpen", BCM_HodOpen);
        fun("BCM_DrvSeatbeltBucklesta", BCM_DrvSeatbeltBucklesta);
        fun("BCM_FrontWiperSt", BCM_FrontWiperSt);
        fun("BCM_HighBeamSt", BCM_HighBeamSt);
        fun("CS1_HighBeamReqSt ", CS1_HighBeamReqSt );
        fun("BCM_LowBeamSt", BCM_LowBeamSt);
        fun("HazardLampSt", HazardLampSt);
        fun("BCM_FrontFogLampSt", BCM_FrontFogLampSt);
        fun("BCM_RearFogLampSt", BCM_RearFogLampSt);
        fun("BCM_LeftTurnLightSt", BCM_LeftTurnLightSt);
        fun("BCM_RightTurnLightSt", BCM_RightTurnLightSt);
        fun("BCM_TurnLightSW", BCM_TurnLightSW);
        fun("BCM_FrontWiperWorkSts", BCM_FrontWiperWorkSts);
        fun("BCM_FrontLampSt", BCM_FrontLampSt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("BCM_FLDrOpn", BCM_FLDrOpn);
        fun("BCM_FRDrOpn", BCM_FRDrOpn);
        fun("BCM_RLDrOpn", BCM_RLDrOpn);
        fun("BCM_RRDrOpn", BCM_RRDrOpn);
        fun("BCM_TGOpn", BCM_TGOpn);
        fun("BCM_HodOpen", BCM_HodOpen);
        fun("BCM_DrvSeatbeltBucklesta", BCM_DrvSeatbeltBucklesta);
        fun("BCM_FrontWiperSt", BCM_FrontWiperSt);
        fun("BCM_HighBeamSt", BCM_HighBeamSt);
        fun("CS1_HighBeamReqSt ", CS1_HighBeamReqSt );
        fun("BCM_LowBeamSt", BCM_LowBeamSt);
        fun("HazardLampSt", HazardLampSt);
        fun("BCM_FrontFogLampSt", BCM_FrontFogLampSt);
        fun("BCM_RearFogLampSt", BCM_RearFogLampSt);
        fun("BCM_LeftTurnLightSt", BCM_LeftTurnLightSt);
        fun("BCM_RightTurnLightSt", BCM_RightTurnLightSt);
        fun("BCM_TurnLightSW", BCM_TurnLightSW);
        fun("BCM_FrontWiperWorkSts", BCM_FrontWiperWorkSts);
        fun("BCM_FrontLampSt", BCM_FrontLampSt);
    }

    bool operator==(const ::hozon::chassis::BodyStateInfo& t) const
    {
        return (BCM_FLDrOpn == t.BCM_FLDrOpn) && (BCM_FRDrOpn == t.BCM_FRDrOpn) && (BCM_RLDrOpn == t.BCM_RLDrOpn) && (BCM_RRDrOpn == t.BCM_RRDrOpn) && (BCM_TGOpn == t.BCM_TGOpn) && (BCM_HodOpen == t.BCM_HodOpen) && (BCM_DrvSeatbeltBucklesta == t.BCM_DrvSeatbeltBucklesta) && (BCM_FrontWiperSt == t.BCM_FrontWiperSt) && (BCM_HighBeamSt == t.BCM_HighBeamSt) && (CS1_HighBeamReqSt  == t.CS1_HighBeamReqSt ) && (BCM_LowBeamSt == t.BCM_LowBeamSt) && (HazardLampSt == t.HazardLampSt) && (BCM_FrontFogLampSt == t.BCM_FrontFogLampSt) && (BCM_RearFogLampSt == t.BCM_RearFogLampSt) && (BCM_LeftTurnLightSt == t.BCM_LeftTurnLightSt) && (BCM_RightTurnLightSt == t.BCM_RightTurnLightSt) && (BCM_TurnLightSW == t.BCM_TurnLightSW) && (BCM_FrontWiperWorkSts == t.BCM_FrontWiperWorkSts) && (BCM_FrontLampSt == t.BCM_FrontLampSt);
    }
};
} // namespace chassis
} // namespace hozon


#endif // HOZON_CHASSIS_IMPL_TYPE_BODYSTATEINFO_H

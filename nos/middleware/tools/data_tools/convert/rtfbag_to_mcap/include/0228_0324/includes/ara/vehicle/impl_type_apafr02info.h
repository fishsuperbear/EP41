/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_APAFR02INFO_H
#define ARA_VEHICLE_IMPL_TYPE_APAFR02INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "ara/vehicle/impl_type_time.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace ara {
namespace vehicle {
struct APAFr02Info {
    ::Boolean valid;
    ::ara::vehicle::Time time;
    ::UInt8 APA_InfoDisplayReq;
    ::Float APA_EPSAngleValueReq;
    ::UInt8 APA_EPSAngleReq;
    ::UInt8 APA_EPSAngleReqValidity;
    ::UInt8 APA_TurnLightReq;
    ::UInt8 APA_SysSoundIndication;
    ::UInt8 APA_WorkSts_CH;
    ::UInt8 APA_Sound_Location;
    ::UInt8 PAS_SoundIndication;
    ::UInt8 APA_ParkBarPercent;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(valid);
        fun(time);
        fun(APA_InfoDisplayReq);
        fun(APA_EPSAngleValueReq);
        fun(APA_EPSAngleReq);
        fun(APA_EPSAngleReqValidity);
        fun(APA_TurnLightReq);
        fun(APA_SysSoundIndication);
        fun(APA_WorkSts_CH);
        fun(APA_Sound_Location);
        fun(PAS_SoundIndication);
        fun(APA_ParkBarPercent);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(valid);
        fun(time);
        fun(APA_InfoDisplayReq);
        fun(APA_EPSAngleValueReq);
        fun(APA_EPSAngleReq);
        fun(APA_EPSAngleReqValidity);
        fun(APA_TurnLightReq);
        fun(APA_SysSoundIndication);
        fun(APA_WorkSts_CH);
        fun(APA_Sound_Location);
        fun(PAS_SoundIndication);
        fun(APA_ParkBarPercent);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("valid", valid);
        fun("time", time);
        fun("APA_InfoDisplayReq", APA_InfoDisplayReq);
        fun("APA_EPSAngleValueReq", APA_EPSAngleValueReq);
        fun("APA_EPSAngleReq", APA_EPSAngleReq);
        fun("APA_EPSAngleReqValidity", APA_EPSAngleReqValidity);
        fun("APA_TurnLightReq", APA_TurnLightReq);
        fun("APA_SysSoundIndication", APA_SysSoundIndication);
        fun("APA_WorkSts_CH", APA_WorkSts_CH);
        fun("APA_Sound_Location", APA_Sound_Location);
        fun("PAS_SoundIndication", PAS_SoundIndication);
        fun("APA_ParkBarPercent", APA_ParkBarPercent);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("valid", valid);
        fun("time", time);
        fun("APA_InfoDisplayReq", APA_InfoDisplayReq);
        fun("APA_EPSAngleValueReq", APA_EPSAngleValueReq);
        fun("APA_EPSAngleReq", APA_EPSAngleReq);
        fun("APA_EPSAngleReqValidity", APA_EPSAngleReqValidity);
        fun("APA_TurnLightReq", APA_TurnLightReq);
        fun("APA_SysSoundIndication", APA_SysSoundIndication);
        fun("APA_WorkSts_CH", APA_WorkSts_CH);
        fun("APA_Sound_Location", APA_Sound_Location);
        fun("PAS_SoundIndication", PAS_SoundIndication);
        fun("APA_ParkBarPercent", APA_ParkBarPercent);
    }

    bool operator==(const ::ara::vehicle::APAFr02Info& t) const
    {
        return (valid == t.valid) && (time == t.time) && (APA_InfoDisplayReq == t.APA_InfoDisplayReq) && (fabs(static_cast<double>(APA_EPSAngleValueReq - t.APA_EPSAngleValueReq)) < DBL_EPSILON) && (APA_EPSAngleReq == t.APA_EPSAngleReq) && (APA_EPSAngleReqValidity == t.APA_EPSAngleReqValidity) && (APA_TurnLightReq == t.APA_TurnLightReq) && (APA_SysSoundIndication == t.APA_SysSoundIndication) && (APA_WorkSts_CH == t.APA_WorkSts_CH) && (APA_Sound_Location == t.APA_Sound_Location) && (PAS_SoundIndication == t.PAS_SoundIndication) && (APA_ParkBarPercent == t.APA_ParkBarPercent);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_APAFR02INFO_H

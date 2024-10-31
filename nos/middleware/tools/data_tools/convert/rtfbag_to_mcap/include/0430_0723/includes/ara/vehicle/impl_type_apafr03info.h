/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_APAFR03INFO_H
#define ARA_VEHICLE_IMPL_TYPE_APAFR03INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "ara/vehicle/impl_type_time.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace ara {
namespace vehicle {
struct APAFr03Info {
    ::Boolean valid;
    ::ara::vehicle::Time time;
    ::UInt8 APA_TorqReq;
    ::UInt8 APA_GearPosReqValidity;
    ::UInt8 APA_GearPosReq;
    ::Float APA_TorqReqValue;
    ::UInt8 APA_TorqReqValidity;

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
        fun(APA_TorqReq);
        fun(APA_GearPosReqValidity);
        fun(APA_GearPosReq);
        fun(APA_TorqReqValue);
        fun(APA_TorqReqValidity);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(valid);
        fun(time);
        fun(APA_TorqReq);
        fun(APA_GearPosReqValidity);
        fun(APA_GearPosReq);
        fun(APA_TorqReqValue);
        fun(APA_TorqReqValidity);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("valid", valid);
        fun("time", time);
        fun("APA_TorqReq", APA_TorqReq);
        fun("APA_GearPosReqValidity", APA_GearPosReqValidity);
        fun("APA_GearPosReq", APA_GearPosReq);
        fun("APA_TorqReqValue", APA_TorqReqValue);
        fun("APA_TorqReqValidity", APA_TorqReqValidity);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("valid", valid);
        fun("time", time);
        fun("APA_TorqReq", APA_TorqReq);
        fun("APA_GearPosReqValidity", APA_GearPosReqValidity);
        fun("APA_GearPosReq", APA_GearPosReq);
        fun("APA_TorqReqValue", APA_TorqReqValue);
        fun("APA_TorqReqValidity", APA_TorqReqValidity);
    }

    bool operator==(const ::ara::vehicle::APAFr03Info& t) const
    {
        return (valid == t.valid) && (time == t.time) && (APA_TorqReq == t.APA_TorqReq) && (APA_GearPosReqValidity == t.APA_GearPosReqValidity) && (APA_GearPosReq == t.APA_GearPosReq) && (fabs(static_cast<double>(APA_TorqReqValue - t.APA_TorqReqValue)) < DBL_EPSILON) && (APA_TorqReqValidity == t.APA_TorqReqValidity);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_APAFR03INFO_H

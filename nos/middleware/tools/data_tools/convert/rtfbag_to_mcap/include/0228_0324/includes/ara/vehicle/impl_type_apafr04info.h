/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_APAFR04INFO_H
#define ARA_VEHICLE_IMPL_TYPE_APAFR04INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "ara/vehicle/impl_type_time.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"
#include "impl_type_double.h"

namespace ara {
namespace vehicle {
struct APAFr04Info {
    ::Boolean valid;
    ::ara::vehicle::Time time;
    ::UInt8 APA_ApaEbpReq;
    ::UInt8 RCW_warningReq;
    ::UInt8 BSD_warningReqRight;
    ::UInt8 BSD_warningReqLeft;
    ::UInt8 BsdSetFd;
    ::UInt8 BSD_SysSts;
    ::Float APA_ApaTarDecel;
    ::Double APA_ApaMaxSpd;
    ::UInt8 APA_ApaBrkMod;
    ::UInt8 APA_ApaMod;
    ::UInt8 APA_ApaStopReq;

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
        fun(APA_ApaEbpReq);
        fun(RCW_warningReq);
        fun(BSD_warningReqRight);
        fun(BSD_warningReqLeft);
        fun(BsdSetFd);
        fun(BSD_SysSts);
        fun(APA_ApaTarDecel);
        fun(APA_ApaMaxSpd);
        fun(APA_ApaBrkMod);
        fun(APA_ApaMod);
        fun(APA_ApaStopReq);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(valid);
        fun(time);
        fun(APA_ApaEbpReq);
        fun(RCW_warningReq);
        fun(BSD_warningReqRight);
        fun(BSD_warningReqLeft);
        fun(BsdSetFd);
        fun(BSD_SysSts);
        fun(APA_ApaTarDecel);
        fun(APA_ApaMaxSpd);
        fun(APA_ApaBrkMod);
        fun(APA_ApaMod);
        fun(APA_ApaStopReq);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("valid", valid);
        fun("time", time);
        fun("APA_ApaEbpReq", APA_ApaEbpReq);
        fun("RCW_warningReq", RCW_warningReq);
        fun("BSD_warningReqRight", BSD_warningReqRight);
        fun("BSD_warningReqLeft", BSD_warningReqLeft);
        fun("BsdSetFd", BsdSetFd);
        fun("BSD_SysSts", BSD_SysSts);
        fun("APA_ApaTarDecel", APA_ApaTarDecel);
        fun("APA_ApaMaxSpd", APA_ApaMaxSpd);
        fun("APA_ApaBrkMod", APA_ApaBrkMod);
        fun("APA_ApaMod", APA_ApaMod);
        fun("APA_ApaStopReq", APA_ApaStopReq);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("valid", valid);
        fun("time", time);
        fun("APA_ApaEbpReq", APA_ApaEbpReq);
        fun("RCW_warningReq", RCW_warningReq);
        fun("BSD_warningReqRight", BSD_warningReqRight);
        fun("BSD_warningReqLeft", BSD_warningReqLeft);
        fun("BsdSetFd", BsdSetFd);
        fun("BSD_SysSts", BSD_SysSts);
        fun("APA_ApaTarDecel", APA_ApaTarDecel);
        fun("APA_ApaMaxSpd", APA_ApaMaxSpd);
        fun("APA_ApaBrkMod", APA_ApaBrkMod);
        fun("APA_ApaMod", APA_ApaMod);
        fun("APA_ApaStopReq", APA_ApaStopReq);
    }

    bool operator==(const ::ara::vehicle::APAFr04Info& t) const
    {
        return (valid == t.valid) && (time == t.time) && (APA_ApaEbpReq == t.APA_ApaEbpReq) && (RCW_warningReq == t.RCW_warningReq) && (BSD_warningReqRight == t.BSD_warningReqRight) && (BSD_warningReqLeft == t.BSD_warningReqLeft) && (BsdSetFd == t.BsdSetFd) && (BSD_SysSts == t.BSD_SysSts) && (fabs(static_cast<double>(APA_ApaTarDecel - t.APA_ApaTarDecel)) < DBL_EPSILON) && (fabs(static_cast<double>(APA_ApaMaxSpd - t.APA_ApaMaxSpd)) < DBL_EPSILON) && (APA_ApaBrkMod == t.APA_ApaBrkMod) && (APA_ApaMod == t.APA_ApaMod) && (APA_ApaStopReq == t.APA_ApaStopReq);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_APAFR04INFO_H

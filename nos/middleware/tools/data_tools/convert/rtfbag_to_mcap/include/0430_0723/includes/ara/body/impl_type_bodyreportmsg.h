/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_BODY_IMPL_TYPE_BODYREPORTMSG_H
#define ARA_BODY_IMPL_TYPE_BODYREPORTMSG_H
#include <cfloat>
#include <cmath>
#include "impl_type_lightswitch.h"
#include "impl_type_modeswitch.h"
#include "impl_type_mirrorswitch.h"
#include "impl_type_wiper.h"
#include "impl_type_padswitch.h"
#include "impl_type_lightstatus.h"
#include "impl_type_doorstatus.h"
#include "impl_type_doorswitch.h"
#include "impl_type_windowstatus.h"
#include "impl_type_tyrestatus.h"
#include "impl_type_safetystatus.h"
#include "impl_type_commonstatus.h"
#include "impl_type_pepsswitch.h"
#include "impl_type_bodyreportheader.h"

namespace ara {
namespace body {
struct BodyReportMsg {
    ::LightSwitch lightSwitch;
    ::ModeSwitch modeSwitch;
    ::MirrorSwitch mirrorSwitch;
    ::Wiper wiperFrontSwitch;
    ::Wiper wiperRearSwitch;
    ::Wiper wiperFrontStatus;
    ::Wiper wiperRearStatus;
    ::PadSwitch padSwitch;
    ::LightStatus lightStatus;
    ::DoorStatus doorStatus;
    ::DoorSwitch doorSwitch;
    ::WindowStatus windowStatus;
    ::TyreStatus tyreFlStatus;
    ::TyreStatus tyreFrStatus;
    ::TyreStatus tyreRlStatus;
    ::TyreStatus tyreRrStatus;
    ::SafetyStatus safetyStatus;
    ::CommonStatus commonStatus;
    ::PepsSwitch pepsSwitch;
    ::BodyReportHeader header;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(lightSwitch);
        fun(modeSwitch);
        fun(mirrorSwitch);
        fun(wiperFrontSwitch);
        fun(wiperRearSwitch);
        fun(wiperFrontStatus);
        fun(wiperRearStatus);
        fun(padSwitch);
        fun(lightStatus);
        fun(doorStatus);
        fun(doorSwitch);
        fun(windowStatus);
        fun(tyreFlStatus);
        fun(tyreFrStatus);
        fun(tyreRlStatus);
        fun(tyreRrStatus);
        fun(safetyStatus);
        fun(commonStatus);
        fun(pepsSwitch);
        fun(header);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(lightSwitch);
        fun(modeSwitch);
        fun(mirrorSwitch);
        fun(wiperFrontSwitch);
        fun(wiperRearSwitch);
        fun(wiperFrontStatus);
        fun(wiperRearStatus);
        fun(padSwitch);
        fun(lightStatus);
        fun(doorStatus);
        fun(doorSwitch);
        fun(windowStatus);
        fun(tyreFlStatus);
        fun(tyreFrStatus);
        fun(tyreRlStatus);
        fun(tyreRrStatus);
        fun(safetyStatus);
        fun(commonStatus);
        fun(pepsSwitch);
        fun(header);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("lightSwitch", lightSwitch);
        fun("modeSwitch", modeSwitch);
        fun("mirrorSwitch", mirrorSwitch);
        fun("wiperFrontSwitch", wiperFrontSwitch);
        fun("wiperRearSwitch", wiperRearSwitch);
        fun("wiperFrontStatus", wiperFrontStatus);
        fun("wiperRearStatus", wiperRearStatus);
        fun("padSwitch", padSwitch);
        fun("lightStatus", lightStatus);
        fun("doorStatus", doorStatus);
        fun("doorSwitch", doorSwitch);
        fun("windowStatus", windowStatus);
        fun("tyreFlStatus", tyreFlStatus);
        fun("tyreFrStatus", tyreFrStatus);
        fun("tyreRlStatus", tyreRlStatus);
        fun("tyreRrStatus", tyreRrStatus);
        fun("safetyStatus", safetyStatus);
        fun("commonStatus", commonStatus);
        fun("pepsSwitch", pepsSwitch);
        fun("header", header);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("lightSwitch", lightSwitch);
        fun("modeSwitch", modeSwitch);
        fun("mirrorSwitch", mirrorSwitch);
        fun("wiperFrontSwitch", wiperFrontSwitch);
        fun("wiperRearSwitch", wiperRearSwitch);
        fun("wiperFrontStatus", wiperFrontStatus);
        fun("wiperRearStatus", wiperRearStatus);
        fun("padSwitch", padSwitch);
        fun("lightStatus", lightStatus);
        fun("doorStatus", doorStatus);
        fun("doorSwitch", doorSwitch);
        fun("windowStatus", windowStatus);
        fun("tyreFlStatus", tyreFlStatus);
        fun("tyreFrStatus", tyreFrStatus);
        fun("tyreRlStatus", tyreRlStatus);
        fun("tyreRrStatus", tyreRrStatus);
        fun("safetyStatus", safetyStatus);
        fun("commonStatus", commonStatus);
        fun("pepsSwitch", pepsSwitch);
        fun("header", header);
    }

    bool operator==(const ::ara::body::BodyReportMsg& t) const
    {
        return (lightSwitch == t.lightSwitch) && (modeSwitch == t.modeSwitch) && (mirrorSwitch == t.mirrorSwitch) && (wiperFrontSwitch == t.wiperFrontSwitch) && (wiperRearSwitch == t.wiperRearSwitch) && (wiperFrontStatus == t.wiperFrontStatus) && (wiperRearStatus == t.wiperRearStatus) && (padSwitch == t.padSwitch) && (lightStatus == t.lightStatus) && (doorStatus == t.doorStatus) && (doorSwitch == t.doorSwitch) && (windowStatus == t.windowStatus) && (tyreFlStatus == t.tyreFlStatus) && (tyreFrStatus == t.tyreFrStatus) && (tyreRlStatus == t.tyreRlStatus) && (tyreRrStatus == t.tyreRrStatus) && (safetyStatus == t.safetyStatus) && (commonStatus == t.commonStatus) && (pepsSwitch == t.pepsSwitch) && (header == t.header);
    }
};
} // namespace body
} // namespace ara


#endif // ARA_BODY_IMPL_TYPE_BODYREPORTMSG_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X8E_H
#define HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X8E_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc2mcu {
struct Adas_MSG_0x8E {
    ::UInt8 ADCS2_ADAS_EPSAngleReqSt;
    ::UInt8 ADCS2_EPS_LDPState;
    ::UInt8 ADCS2_NNP_SystemFaultStatus;
    ::UInt8 ADCS_Pilot_SystemFaultStatus;
    ::UInt8 ADCS2_ADAS_EPSLateralCtrlType;
    ::UInt8 ADCS2_ADSDriving_mode;
    ::Float ADCS2_ADAS_EPSAngleReq;
    ::UInt8 ADCS2_ALCDirection;
    ::UInt8 ADCS2_EPS_ELKState;
    ::UInt8 ADCS2_longitudCtrlType;
    ::UInt8 ADCS2_LongitudCtrlDecToStopReq;
    ::UInt8 ADCS2_LongitudCtrlDriveOff;
    ::UInt8 ADCS2_LongitudCtrlAccelCtrlReq;
    ::Float ADCS2_LongitudCtrlTargetAccel;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS2_ADAS_EPSAngleReqSt);
        fun(ADCS2_EPS_LDPState);
        fun(ADCS2_NNP_SystemFaultStatus);
        fun(ADCS_Pilot_SystemFaultStatus);
        fun(ADCS2_ADAS_EPSLateralCtrlType);
        fun(ADCS2_ADSDriving_mode);
        fun(ADCS2_ADAS_EPSAngleReq);
        fun(ADCS2_ALCDirection);
        fun(ADCS2_EPS_ELKState);
        fun(ADCS2_longitudCtrlType);
        fun(ADCS2_LongitudCtrlDecToStopReq);
        fun(ADCS2_LongitudCtrlDriveOff);
        fun(ADCS2_LongitudCtrlAccelCtrlReq);
        fun(ADCS2_LongitudCtrlTargetAccel);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS2_ADAS_EPSAngleReqSt);
        fun(ADCS2_EPS_LDPState);
        fun(ADCS2_NNP_SystemFaultStatus);
        fun(ADCS_Pilot_SystemFaultStatus);
        fun(ADCS2_ADAS_EPSLateralCtrlType);
        fun(ADCS2_ADSDriving_mode);
        fun(ADCS2_ADAS_EPSAngleReq);
        fun(ADCS2_ALCDirection);
        fun(ADCS2_EPS_ELKState);
        fun(ADCS2_longitudCtrlType);
        fun(ADCS2_LongitudCtrlDecToStopReq);
        fun(ADCS2_LongitudCtrlDriveOff);
        fun(ADCS2_LongitudCtrlAccelCtrlReq);
        fun(ADCS2_LongitudCtrlTargetAccel);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS2_ADAS_EPSAngleReqSt", ADCS2_ADAS_EPSAngleReqSt);
        fun("ADCS2_EPS_LDPState", ADCS2_EPS_LDPState);
        fun("ADCS2_NNP_SystemFaultStatus", ADCS2_NNP_SystemFaultStatus);
        fun("ADCS_Pilot_SystemFaultStatus", ADCS_Pilot_SystemFaultStatus);
        fun("ADCS2_ADAS_EPSLateralCtrlType", ADCS2_ADAS_EPSLateralCtrlType);
        fun("ADCS2_ADSDriving_mode", ADCS2_ADSDriving_mode);
        fun("ADCS2_ADAS_EPSAngleReq", ADCS2_ADAS_EPSAngleReq);
        fun("ADCS2_ALCDirection", ADCS2_ALCDirection);
        fun("ADCS2_EPS_ELKState", ADCS2_EPS_ELKState);
        fun("ADCS2_longitudCtrlType", ADCS2_longitudCtrlType);
        fun("ADCS2_LongitudCtrlDecToStopReq", ADCS2_LongitudCtrlDecToStopReq);
        fun("ADCS2_LongitudCtrlDriveOff", ADCS2_LongitudCtrlDriveOff);
        fun("ADCS2_LongitudCtrlAccelCtrlReq", ADCS2_LongitudCtrlAccelCtrlReq);
        fun("ADCS2_LongitudCtrlTargetAccel", ADCS2_LongitudCtrlTargetAccel);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS2_ADAS_EPSAngleReqSt", ADCS2_ADAS_EPSAngleReqSt);
        fun("ADCS2_EPS_LDPState", ADCS2_EPS_LDPState);
        fun("ADCS2_NNP_SystemFaultStatus", ADCS2_NNP_SystemFaultStatus);
        fun("ADCS_Pilot_SystemFaultStatus", ADCS_Pilot_SystemFaultStatus);
        fun("ADCS2_ADAS_EPSLateralCtrlType", ADCS2_ADAS_EPSLateralCtrlType);
        fun("ADCS2_ADSDriving_mode", ADCS2_ADSDriving_mode);
        fun("ADCS2_ADAS_EPSAngleReq", ADCS2_ADAS_EPSAngleReq);
        fun("ADCS2_ALCDirection", ADCS2_ALCDirection);
        fun("ADCS2_EPS_ELKState", ADCS2_EPS_ELKState);
        fun("ADCS2_longitudCtrlType", ADCS2_longitudCtrlType);
        fun("ADCS2_LongitudCtrlDecToStopReq", ADCS2_LongitudCtrlDecToStopReq);
        fun("ADCS2_LongitudCtrlDriveOff", ADCS2_LongitudCtrlDriveOff);
        fun("ADCS2_LongitudCtrlAccelCtrlReq", ADCS2_LongitudCtrlAccelCtrlReq);
        fun("ADCS2_LongitudCtrlTargetAccel", ADCS2_LongitudCtrlTargetAccel);
    }

    bool operator==(const ::hozon::soc2mcu::Adas_MSG_0x8E& t) const
    {
        return (ADCS2_ADAS_EPSAngleReqSt == t.ADCS2_ADAS_EPSAngleReqSt) && (ADCS2_EPS_LDPState == t.ADCS2_EPS_LDPState) && (ADCS2_NNP_SystemFaultStatus == t.ADCS2_NNP_SystemFaultStatus) && (ADCS_Pilot_SystemFaultStatus == t.ADCS_Pilot_SystemFaultStatus) && (ADCS2_ADAS_EPSLateralCtrlType == t.ADCS2_ADAS_EPSLateralCtrlType) && (ADCS2_ADSDriving_mode == t.ADCS2_ADSDriving_mode) && (fabs(static_cast<double>(ADCS2_ADAS_EPSAngleReq - t.ADCS2_ADAS_EPSAngleReq)) < DBL_EPSILON) && (ADCS2_ALCDirection == t.ADCS2_ALCDirection) && (ADCS2_EPS_ELKState == t.ADCS2_EPS_ELKState) && (ADCS2_longitudCtrlType == t.ADCS2_longitudCtrlType) && (ADCS2_LongitudCtrlDecToStopReq == t.ADCS2_LongitudCtrlDecToStopReq) && (ADCS2_LongitudCtrlDriveOff == t.ADCS2_LongitudCtrlDriveOff) && (ADCS2_LongitudCtrlAccelCtrlReq == t.ADCS2_LongitudCtrlAccelCtrlReq) && (fabs(static_cast<double>(ADCS2_LongitudCtrlTargetAccel - t.ADCS2_LongitudCtrlTargetAccel)) < DBL_EPSILON);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_ADAS_MSG_0X8E_H

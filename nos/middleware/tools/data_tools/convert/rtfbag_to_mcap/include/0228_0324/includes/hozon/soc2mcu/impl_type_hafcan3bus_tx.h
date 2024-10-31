/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFCAN3BUS_TX_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFCAN3BUS_TX_H
#include <cfloat>
#include <cmath>
#include "impl_type_boolean.h"
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace hozon {
namespace soc2mcu {
struct HafCan3Bus_Tx {
    ::Boolean ADCS2_AEBVehicleHoldReq;
    ::Boolean ADCS2_AEBFullBrake;
    ::Boolean ADCS2_AEBPartialBrake;
    ::UInt8 ADCS2_AEB_DBSLevel;
    ::Boolean ADCS2_AEBPrefillReq;
    ::Boolean ADCS2_AEB_JerkReq;
    ::Float ADCS2_AEBTargetDeceleration;
    ::UInt8 ADCS2_AEB_JerkLevel;
    ::UInt8 ADCS8_AEB_SystemFaultStatus;
    ::Boolean ADCS8_AEB_SystemStatus;
    ::UInt8 ADCS8_FCWSystemFaultStatus;
    ::UInt8 ADCS8_FCWStatus;
    ::UInt8 ADCS8_FCWSensitiveLevel;
    ::Boolean ADCS8_FCW_State;
    ::UInt8 ADCS8_SystemFault;
    ::Boolean ADCS12_FCWWarnAudioplay;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS2_AEBVehicleHoldReq);
        fun(ADCS2_AEBFullBrake);
        fun(ADCS2_AEBPartialBrake);
        fun(ADCS2_AEB_DBSLevel);
        fun(ADCS2_AEBPrefillReq);
        fun(ADCS2_AEB_JerkReq);
        fun(ADCS2_AEBTargetDeceleration);
        fun(ADCS2_AEB_JerkLevel);
        fun(ADCS8_AEB_SystemFaultStatus);
        fun(ADCS8_AEB_SystemStatus);
        fun(ADCS8_FCWSystemFaultStatus);
        fun(ADCS8_FCWStatus);
        fun(ADCS8_FCWSensitiveLevel);
        fun(ADCS8_FCW_State);
        fun(ADCS8_SystemFault);
        fun(ADCS12_FCWWarnAudioplay);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS2_AEBVehicleHoldReq);
        fun(ADCS2_AEBFullBrake);
        fun(ADCS2_AEBPartialBrake);
        fun(ADCS2_AEB_DBSLevel);
        fun(ADCS2_AEBPrefillReq);
        fun(ADCS2_AEB_JerkReq);
        fun(ADCS2_AEBTargetDeceleration);
        fun(ADCS2_AEB_JerkLevel);
        fun(ADCS8_AEB_SystemFaultStatus);
        fun(ADCS8_AEB_SystemStatus);
        fun(ADCS8_FCWSystemFaultStatus);
        fun(ADCS8_FCWStatus);
        fun(ADCS8_FCWSensitiveLevel);
        fun(ADCS8_FCW_State);
        fun(ADCS8_SystemFault);
        fun(ADCS12_FCWWarnAudioplay);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS2_AEBVehicleHoldReq", ADCS2_AEBVehicleHoldReq);
        fun("ADCS2_AEBFullBrake", ADCS2_AEBFullBrake);
        fun("ADCS2_AEBPartialBrake", ADCS2_AEBPartialBrake);
        fun("ADCS2_AEB_DBSLevel", ADCS2_AEB_DBSLevel);
        fun("ADCS2_AEBPrefillReq", ADCS2_AEBPrefillReq);
        fun("ADCS2_AEB_JerkReq", ADCS2_AEB_JerkReq);
        fun("ADCS2_AEBTargetDeceleration", ADCS2_AEBTargetDeceleration);
        fun("ADCS2_AEB_JerkLevel", ADCS2_AEB_JerkLevel);
        fun("ADCS8_AEB_SystemFaultStatus", ADCS8_AEB_SystemFaultStatus);
        fun("ADCS8_AEB_SystemStatus", ADCS8_AEB_SystemStatus);
        fun("ADCS8_FCWSystemFaultStatus", ADCS8_FCWSystemFaultStatus);
        fun("ADCS8_FCWStatus", ADCS8_FCWStatus);
        fun("ADCS8_FCWSensitiveLevel", ADCS8_FCWSensitiveLevel);
        fun("ADCS8_FCW_State", ADCS8_FCW_State);
        fun("ADCS8_SystemFault", ADCS8_SystemFault);
        fun("ADCS12_FCWWarnAudioplay", ADCS12_FCWWarnAudioplay);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS2_AEBVehicleHoldReq", ADCS2_AEBVehicleHoldReq);
        fun("ADCS2_AEBFullBrake", ADCS2_AEBFullBrake);
        fun("ADCS2_AEBPartialBrake", ADCS2_AEBPartialBrake);
        fun("ADCS2_AEB_DBSLevel", ADCS2_AEB_DBSLevel);
        fun("ADCS2_AEBPrefillReq", ADCS2_AEBPrefillReq);
        fun("ADCS2_AEB_JerkReq", ADCS2_AEB_JerkReq);
        fun("ADCS2_AEBTargetDeceleration", ADCS2_AEBTargetDeceleration);
        fun("ADCS2_AEB_JerkLevel", ADCS2_AEB_JerkLevel);
        fun("ADCS8_AEB_SystemFaultStatus", ADCS8_AEB_SystemFaultStatus);
        fun("ADCS8_AEB_SystemStatus", ADCS8_AEB_SystemStatus);
        fun("ADCS8_FCWSystemFaultStatus", ADCS8_FCWSystemFaultStatus);
        fun("ADCS8_FCWStatus", ADCS8_FCWStatus);
        fun("ADCS8_FCWSensitiveLevel", ADCS8_FCWSensitiveLevel);
        fun("ADCS8_FCW_State", ADCS8_FCW_State);
        fun("ADCS8_SystemFault", ADCS8_SystemFault);
        fun("ADCS12_FCWWarnAudioplay", ADCS12_FCWWarnAudioplay);
    }

    bool operator==(const ::hozon::soc2mcu::HafCan3Bus_Tx& t) const
    {
        return (ADCS2_AEBVehicleHoldReq == t.ADCS2_AEBVehicleHoldReq) && (ADCS2_AEBFullBrake == t.ADCS2_AEBFullBrake) && (ADCS2_AEBPartialBrake == t.ADCS2_AEBPartialBrake) && (ADCS2_AEB_DBSLevel == t.ADCS2_AEB_DBSLevel) && (ADCS2_AEBPrefillReq == t.ADCS2_AEBPrefillReq) && (ADCS2_AEB_JerkReq == t.ADCS2_AEB_JerkReq) && (fabs(static_cast<double>(ADCS2_AEBTargetDeceleration - t.ADCS2_AEBTargetDeceleration)) < DBL_EPSILON) && (ADCS2_AEB_JerkLevel == t.ADCS2_AEB_JerkLevel) && (ADCS8_AEB_SystemFaultStatus == t.ADCS8_AEB_SystemFaultStatus) && (ADCS8_AEB_SystemStatus == t.ADCS8_AEB_SystemStatus) && (ADCS8_FCWSystemFaultStatus == t.ADCS8_FCWSystemFaultStatus) && (ADCS8_FCWStatus == t.ADCS8_FCWStatus) && (ADCS8_FCWSensitiveLevel == t.ADCS8_FCWSensitiveLevel) && (ADCS8_FCW_State == t.ADCS8_FCW_State) && (ADCS8_SystemFault == t.ADCS8_SystemFault) && (ADCS12_FCWWarnAudioplay == t.ADCS12_FCWWarnAudioplay);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFCAN3BUS_TX_H

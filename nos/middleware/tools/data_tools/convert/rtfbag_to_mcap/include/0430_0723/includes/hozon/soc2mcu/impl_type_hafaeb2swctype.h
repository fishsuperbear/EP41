/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_HAFAEB2SWCTYPE_H
#define HOZON_SOC2MCU_IMPL_TYPE_HAFAEB2SWCTYPE_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_boolean.h"

namespace hozon {
namespace soc2mcu {
struct HafAEB2SwcType {
    ::UInt8 ADCS_AEB_Triggering_object_ID;
    ::UInt8 ADCS_FCW_Triggering_object_ID;
    ::UInt8 ADCS2_AEB_JerkLevel;
    ::UInt8 ADCS8_FCWStatus;
    ::Boolean Fusa_MonitorFault;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ADCS_AEB_Triggering_object_ID);
        fun(ADCS_FCW_Triggering_object_ID);
        fun(ADCS2_AEB_JerkLevel);
        fun(ADCS8_FCWStatus);
        fun(Fusa_MonitorFault);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ADCS_AEB_Triggering_object_ID);
        fun(ADCS_FCW_Triggering_object_ID);
        fun(ADCS2_AEB_JerkLevel);
        fun(ADCS8_FCWStatus);
        fun(Fusa_MonitorFault);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("ADCS_AEB_Triggering_object_ID", ADCS_AEB_Triggering_object_ID);
        fun("ADCS_FCW_Triggering_object_ID", ADCS_FCW_Triggering_object_ID);
        fun("ADCS2_AEB_JerkLevel", ADCS2_AEB_JerkLevel);
        fun("ADCS8_FCWStatus", ADCS8_FCWStatus);
        fun("Fusa_MonitorFault", Fusa_MonitorFault);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("ADCS_AEB_Triggering_object_ID", ADCS_AEB_Triggering_object_ID);
        fun("ADCS_FCW_Triggering_object_ID", ADCS_FCW_Triggering_object_ID);
        fun("ADCS2_AEB_JerkLevel", ADCS2_AEB_JerkLevel);
        fun("ADCS8_FCWStatus", ADCS8_FCWStatus);
        fun("Fusa_MonitorFault", Fusa_MonitorFault);
    }

    bool operator==(const ::hozon::soc2mcu::HafAEB2SwcType& t) const
    {
        return (ADCS_AEB_Triggering_object_ID == t.ADCS_AEB_Triggering_object_ID) && (ADCS_FCW_Triggering_object_ID == t.ADCS_FCW_Triggering_object_ID) && (ADCS2_AEB_JerkLevel == t.ADCS2_AEB_JerkLevel) && (ADCS8_FCWStatus == t.ADCS8_FCWStatus) && (Fusa_MonitorFault == t.Fusa_MonitorFault);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_HAFAEB2SWCTYPE_H

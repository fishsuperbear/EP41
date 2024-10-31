/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_ECULOSTSTATUS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_ECULOSTSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_INTF_EcuLostStatus {
    ::UInt8 INTF_Ecu_Lost_ACU;
    ::UInt8 INTF_Ecu_Lost_BDCS;
    ::UInt8 INTF_Ecu_Lost_BTM;
    ::UInt8 INTF_Ecu_Lost_CDCS;
    ::UInt8 INTF_Ecu_Lost_DDCU;
    ::UInt8 INTF_Ecu_Lost_EDU;
    ::UInt8 INTF_Ecu_Lost_EPS;
    ::UInt8 INTF_Ecu_Lost_FMCU;
    ::UInt8 INTF_Ecu_Lost_GW;
    ::UInt8 INTF_Ecu_Lost_ICU;
    ::UInt8 INTF_Ecu_Lost_IDB;
    ::UInt8 INTF_Ecu_Lost_MCU;
    ::UInt8 INTF_Ecu_Lost_PDCU;
    ::UInt8 INTF_Ecu_Lost_RCU;
    ::UInt8 INTF_Ecu_Lost_TBOX;
    ::UInt8 INTF_Ecu_Lost_PDCS;
    ::UInt8 INTF_Ecu_Lost_SOC;
    ::UInt8 INTF_Ecu_Lost_BMS;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTF_Ecu_Lost_ACU);
        fun(INTF_Ecu_Lost_BDCS);
        fun(INTF_Ecu_Lost_BTM);
        fun(INTF_Ecu_Lost_CDCS);
        fun(INTF_Ecu_Lost_DDCU);
        fun(INTF_Ecu_Lost_EDU);
        fun(INTF_Ecu_Lost_EPS);
        fun(INTF_Ecu_Lost_FMCU);
        fun(INTF_Ecu_Lost_GW);
        fun(INTF_Ecu_Lost_ICU);
        fun(INTF_Ecu_Lost_IDB);
        fun(INTF_Ecu_Lost_MCU);
        fun(INTF_Ecu_Lost_PDCU);
        fun(INTF_Ecu_Lost_RCU);
        fun(INTF_Ecu_Lost_TBOX);
        fun(INTF_Ecu_Lost_PDCS);
        fun(INTF_Ecu_Lost_SOC);
        fun(INTF_Ecu_Lost_BMS);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTF_Ecu_Lost_ACU);
        fun(INTF_Ecu_Lost_BDCS);
        fun(INTF_Ecu_Lost_BTM);
        fun(INTF_Ecu_Lost_CDCS);
        fun(INTF_Ecu_Lost_DDCU);
        fun(INTF_Ecu_Lost_EDU);
        fun(INTF_Ecu_Lost_EPS);
        fun(INTF_Ecu_Lost_FMCU);
        fun(INTF_Ecu_Lost_GW);
        fun(INTF_Ecu_Lost_ICU);
        fun(INTF_Ecu_Lost_IDB);
        fun(INTF_Ecu_Lost_MCU);
        fun(INTF_Ecu_Lost_PDCU);
        fun(INTF_Ecu_Lost_RCU);
        fun(INTF_Ecu_Lost_TBOX);
        fun(INTF_Ecu_Lost_PDCS);
        fun(INTF_Ecu_Lost_SOC);
        fun(INTF_Ecu_Lost_BMS);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTF_Ecu_Lost_ACU", INTF_Ecu_Lost_ACU);
        fun("INTF_Ecu_Lost_BDCS", INTF_Ecu_Lost_BDCS);
        fun("INTF_Ecu_Lost_BTM", INTF_Ecu_Lost_BTM);
        fun("INTF_Ecu_Lost_CDCS", INTF_Ecu_Lost_CDCS);
        fun("INTF_Ecu_Lost_DDCU", INTF_Ecu_Lost_DDCU);
        fun("INTF_Ecu_Lost_EDU", INTF_Ecu_Lost_EDU);
        fun("INTF_Ecu_Lost_EPS", INTF_Ecu_Lost_EPS);
        fun("INTF_Ecu_Lost_FMCU", INTF_Ecu_Lost_FMCU);
        fun("INTF_Ecu_Lost_GW", INTF_Ecu_Lost_GW);
        fun("INTF_Ecu_Lost_ICU", INTF_Ecu_Lost_ICU);
        fun("INTF_Ecu_Lost_IDB", INTF_Ecu_Lost_IDB);
        fun("INTF_Ecu_Lost_MCU", INTF_Ecu_Lost_MCU);
        fun("INTF_Ecu_Lost_PDCU", INTF_Ecu_Lost_PDCU);
        fun("INTF_Ecu_Lost_RCU", INTF_Ecu_Lost_RCU);
        fun("INTF_Ecu_Lost_TBOX", INTF_Ecu_Lost_TBOX);
        fun("INTF_Ecu_Lost_PDCS", INTF_Ecu_Lost_PDCS);
        fun("INTF_Ecu_Lost_SOC", INTF_Ecu_Lost_SOC);
        fun("INTF_Ecu_Lost_BMS", INTF_Ecu_Lost_BMS);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTF_Ecu_Lost_ACU", INTF_Ecu_Lost_ACU);
        fun("INTF_Ecu_Lost_BDCS", INTF_Ecu_Lost_BDCS);
        fun("INTF_Ecu_Lost_BTM", INTF_Ecu_Lost_BTM);
        fun("INTF_Ecu_Lost_CDCS", INTF_Ecu_Lost_CDCS);
        fun("INTF_Ecu_Lost_DDCU", INTF_Ecu_Lost_DDCU);
        fun("INTF_Ecu_Lost_EDU", INTF_Ecu_Lost_EDU);
        fun("INTF_Ecu_Lost_EPS", INTF_Ecu_Lost_EPS);
        fun("INTF_Ecu_Lost_FMCU", INTF_Ecu_Lost_FMCU);
        fun("INTF_Ecu_Lost_GW", INTF_Ecu_Lost_GW);
        fun("INTF_Ecu_Lost_ICU", INTF_Ecu_Lost_ICU);
        fun("INTF_Ecu_Lost_IDB", INTF_Ecu_Lost_IDB);
        fun("INTF_Ecu_Lost_MCU", INTF_Ecu_Lost_MCU);
        fun("INTF_Ecu_Lost_PDCU", INTF_Ecu_Lost_PDCU);
        fun("INTF_Ecu_Lost_RCU", INTF_Ecu_Lost_RCU);
        fun("INTF_Ecu_Lost_TBOX", INTF_Ecu_Lost_TBOX);
        fun("INTF_Ecu_Lost_PDCS", INTF_Ecu_Lost_PDCS);
        fun("INTF_Ecu_Lost_SOC", INTF_Ecu_Lost_SOC);
        fun("INTF_Ecu_Lost_BMS", INTF_Ecu_Lost_BMS);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_INTF_EcuLostStatus& t) const
    {
        return (INTF_Ecu_Lost_ACU == t.INTF_Ecu_Lost_ACU) && (INTF_Ecu_Lost_BDCS == t.INTF_Ecu_Lost_BDCS) && (INTF_Ecu_Lost_BTM == t.INTF_Ecu_Lost_BTM) && (INTF_Ecu_Lost_CDCS == t.INTF_Ecu_Lost_CDCS) && (INTF_Ecu_Lost_DDCU == t.INTF_Ecu_Lost_DDCU) && (INTF_Ecu_Lost_EDU == t.INTF_Ecu_Lost_EDU) && (INTF_Ecu_Lost_EPS == t.INTF_Ecu_Lost_EPS) && (INTF_Ecu_Lost_FMCU == t.INTF_Ecu_Lost_FMCU) && (INTF_Ecu_Lost_GW == t.INTF_Ecu_Lost_GW) && (INTF_Ecu_Lost_ICU == t.INTF_Ecu_Lost_ICU) && (INTF_Ecu_Lost_IDB == t.INTF_Ecu_Lost_IDB) && (INTF_Ecu_Lost_MCU == t.INTF_Ecu_Lost_MCU) && (INTF_Ecu_Lost_PDCU == t.INTF_Ecu_Lost_PDCU) && (INTF_Ecu_Lost_RCU == t.INTF_Ecu_Lost_RCU) && (INTF_Ecu_Lost_TBOX == t.INTF_Ecu_Lost_TBOX) && (INTF_Ecu_Lost_PDCS == t.INTF_Ecu_Lost_PDCS) && (INTF_Ecu_Lost_SOC == t.INTF_Ecu_Lost_SOC) && (INTF_Ecu_Lost_BMS == t.INTF_Ecu_Lost_BMS);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_INTF_ECULOSTSTATUS_H

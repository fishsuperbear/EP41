/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_DATATRANSMITSTATUS_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_DATATRANSMITSTATUS_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint16.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_INTF_DataTransmitStatus {
    ::UInt16 INTF_Data_Adas_HafLaneDetectionOutdata_Cnt;
    ::UInt16 INTF_Data_Adas_HafFusionOutdata_Cnt;
    ::UInt16 INTF_Data_Adas_HafLocationdata_Cnt;
    ::UInt16 INTF_Data_Adas_HafEgoTrajectorydata_Cnt;
    ::UInt16 INTF_Data_Adas_AdptrIn_SOC_Cnt;
    ::UInt16 INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt;
    ::UInt16 INTF_Data_Adas_HafChassis_Cnt;
    ::UInt16 INTF_Data_Adas_HafGlobalTimedata_Cnt;
    ::UInt16 INTF_Data_Adas_PwrCurState_Cnt;
    ::UInt16 INTF_Data_HM_Cnt;
    ::UInt16 INTF_Data_FM_Cnt;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTF_Data_Adas_HafLaneDetectionOutdata_Cnt);
        fun(INTF_Data_Adas_HafFusionOutdata_Cnt);
        fun(INTF_Data_Adas_HafLocationdata_Cnt);
        fun(INTF_Data_Adas_HafEgoTrajectorydata_Cnt);
        fun(INTF_Data_Adas_AdptrIn_SOC_Cnt);
        fun(INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt);
        fun(INTF_Data_Adas_HafChassis_Cnt);
        fun(INTF_Data_Adas_HafGlobalTimedata_Cnt);
        fun(INTF_Data_Adas_PwrCurState_Cnt);
        fun(INTF_Data_HM_Cnt);
        fun(INTF_Data_FM_Cnt);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTF_Data_Adas_HafLaneDetectionOutdata_Cnt);
        fun(INTF_Data_Adas_HafFusionOutdata_Cnt);
        fun(INTF_Data_Adas_HafLocationdata_Cnt);
        fun(INTF_Data_Adas_HafEgoTrajectorydata_Cnt);
        fun(INTF_Data_Adas_AdptrIn_SOC_Cnt);
        fun(INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt);
        fun(INTF_Data_Adas_HafChassis_Cnt);
        fun(INTF_Data_Adas_HafGlobalTimedata_Cnt);
        fun(INTF_Data_Adas_PwrCurState_Cnt);
        fun(INTF_Data_HM_Cnt);
        fun(INTF_Data_FM_Cnt);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTF_Data_Adas_HafLaneDetectionOutdata_Cnt", INTF_Data_Adas_HafLaneDetectionOutdata_Cnt);
        fun("INTF_Data_Adas_HafFusionOutdata_Cnt", INTF_Data_Adas_HafFusionOutdata_Cnt);
        fun("INTF_Data_Adas_HafLocationdata_Cnt", INTF_Data_Adas_HafLocationdata_Cnt);
        fun("INTF_Data_Adas_HafEgoTrajectorydata_Cnt", INTF_Data_Adas_HafEgoTrajectorydata_Cnt);
        fun("INTF_Data_Adas_AdptrIn_SOC_Cnt", INTF_Data_Adas_AdptrIn_SOC_Cnt);
        fun("INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt", INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt);
        fun("INTF_Data_Adas_HafChassis_Cnt", INTF_Data_Adas_HafChassis_Cnt);
        fun("INTF_Data_Adas_HafGlobalTimedata_Cnt", INTF_Data_Adas_HafGlobalTimedata_Cnt);
        fun("INTF_Data_Adas_PwrCurState_Cnt", INTF_Data_Adas_PwrCurState_Cnt);
        fun("INTF_Data_HM_Cnt", INTF_Data_HM_Cnt);
        fun("INTF_Data_FM_Cnt", INTF_Data_FM_Cnt);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTF_Data_Adas_HafLaneDetectionOutdata_Cnt", INTF_Data_Adas_HafLaneDetectionOutdata_Cnt);
        fun("INTF_Data_Adas_HafFusionOutdata_Cnt", INTF_Data_Adas_HafFusionOutdata_Cnt);
        fun("INTF_Data_Adas_HafLocationdata_Cnt", INTF_Data_Adas_HafLocationdata_Cnt);
        fun("INTF_Data_Adas_HafEgoTrajectorydata_Cnt", INTF_Data_Adas_HafEgoTrajectorydata_Cnt);
        fun("INTF_Data_Adas_AdptrIn_SOC_Cnt", INTF_Data_Adas_AdptrIn_SOC_Cnt);
        fun("INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt", INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt);
        fun("INTF_Data_Adas_HafChassis_Cnt", INTF_Data_Adas_HafChassis_Cnt);
        fun("INTF_Data_Adas_HafGlobalTimedata_Cnt", INTF_Data_Adas_HafGlobalTimedata_Cnt);
        fun("INTF_Data_Adas_PwrCurState_Cnt", INTF_Data_Adas_PwrCurState_Cnt);
        fun("INTF_Data_HM_Cnt", INTF_Data_HM_Cnt);
        fun("INTF_Data_FM_Cnt", INTF_Data_FM_Cnt);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_INTF_DataTransmitStatus& t) const
    {
        return (INTF_Data_Adas_HafLaneDetectionOutdata_Cnt == t.INTF_Data_Adas_HafLaneDetectionOutdata_Cnt) && (INTF_Data_Adas_HafFusionOutdata_Cnt == t.INTF_Data_Adas_HafFusionOutdata_Cnt) && (INTF_Data_Adas_HafLocationdata_Cnt == t.INTF_Data_Adas_HafLocationdata_Cnt) && (INTF_Data_Adas_HafEgoTrajectorydata_Cnt == t.INTF_Data_Adas_HafEgoTrajectorydata_Cnt) && (INTF_Data_Adas_AdptrIn_SOC_Cnt == t.INTF_Data_Adas_AdptrIn_SOC_Cnt) && (INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt == t.INTF_Data_Adas_VEH_CAN_Inputs_EP40_Cnt) && (INTF_Data_Adas_HafChassis_Cnt == t.INTF_Data_Adas_HafChassis_Cnt) && (INTF_Data_Adas_HafGlobalTimedata_Cnt == t.INTF_Data_Adas_HafGlobalTimedata_Cnt) && (INTF_Data_Adas_PwrCurState_Cnt == t.INTF_Data_Adas_PwrCurState_Cnt) && (INTF_Data_HM_Cnt == t.INTF_Data_HM_Cnt) && (INTF_Data_FM_Cnt == t.INTF_Data_FM_Cnt);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_DATATRANSMITSTATUS_H

/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file impl_type_dtcloud_cm_datatransmitstatus.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_DATATRANSMITSTATUS_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_DATATRANSMITSTATUS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_CM_DataTransmitStatus {
    std::uint16_t CM_Data_Adas_HafLaneDetectionOutdata_Cnt;
    std::uint16_t CM_Data_Adas_HafFusionOutdata_Cnt;
    std::uint16_t CM_Data_Adas_HafLocationdata_Cnt;
    std::uint16_t CM_Data_Adas_HafEgoTrajectorydata_Cnt;
    std::uint16_t CM_Data_Adas_AdptrIn_SOC_Cnt;
    std::uint16_t CM_Data_Adas_VEH_CAN_Inputs_EP40_Cnt;
    std::uint16_t CM_Data_Adas_HafChassis_Cnt;
    std::uint16_t CM_Data_Adas_HafGlobalTimedata_Cnt;
    std::uint16_t CM_Data_Adas_PwrCurState_Cnt;
    std::uint16_t CM_Data_HM_Cnt;
    std::uint16_t CM_Data_FM_Cnt;
    std::uint16_t CANBus_Data_ETH_SRRFL_cnt;
    std::uint16_t CANBus_Data_ETH_SRRFR_cnt;
    std::uint16_t CANBus_Data_ETH_SRRRL_cnt;
    std::uint16_t CANBus_Data_ETH_SRRRR_cnt;
    std::uint16_t CANBus_Data_ETH_ARS_cnt;
    std::uint16_t CANBus_Data_ETH_AlgGnssInfo_cnt;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_CM_DataTransmitStatus,CM_Data_Adas_HafLaneDetectionOutdata_Cnt,CM_Data_Adas_HafFusionOutdata_Cnt,CM_Data_Adas_HafLocationdata_Cnt,CM_Data_Adas_HafEgoTrajectorydata_Cnt,CM_Data_Adas_AdptrIn_SOC_Cnt,CM_Data_Adas_VEH_CAN_Inputs_EP40_Cnt,CM_Data_Adas_HafChassis_Cnt,CM_Data_Adas_HafGlobalTimedata_Cnt,CM_Data_Adas_PwrCurState_Cnt,CM_Data_HM_Cnt,CM_Data_FM_Cnt,CANBus_Data_ETH_SRRFL_cnt,CANBus_Data_ETH_SRRFR_cnt,CANBus_Data_ETH_SRRRL_cnt,CANBus_Data_ETH_SRRRR_cnt,CANBus_Data_ETH_ARS_cnt,CANBus_Data_ETH_AlgGnssInfo_cnt);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_DATATRANSMITSTATUS_H_
/* EOF */
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
 * @file impl_type_canfd_msg194.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_CANFD_MSG194_H_
#define HOZON_NETAOS_IMPL_TYPE_CANFD_MSG194_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct CANFD_Msg194 {
    float ADCS5_RPA_slot_ID_9_P0_X;
    float ADCS5_RPA_slot_ID_9_P0_Y;
    float ADCS5_RPA_slot_ID_9_P1_X;
    float ADCS5_RPA_slot_ID_9_P1_Y;
    float ADCS5_RPA_slot_ID_9_P2_X;
    float ADCS5_RPA_slot_ID_9_P2_Y;
    float ADCS5_RPA_slot_ID_9_P3_X;
    float ADCS5_RPA_slot_ID_9_P3_Y;
    float ADCS5_RPA_slot_Angle9;
    float ADCS5_RPA_ParkingSlotDepth9;
    float ADCS5_RPA_ParkingSlotWidth9;
    std::uint8_t ADCS5_RPA_slot_ID_9_Status;
    std::uint8_t ADCS5_RPA_ParkingSlotType9;
    std::uint8_t ADCS5_RPA_ParkingSlotDirection9;
    std::uint8_t ADCS8_PA_warninginfo;
    std::uint8_t ADCS8_PA_FPAS_SensorFaultStsFRC;
    std::uint8_t ADCS8_PA_FPAS_SensorFaultStsFRM;
    std::uint8_t ADCS8_PA_FPAS_SensorFaultStsFLM;
    std::uint8_t ADCS8_PA_FPAS_SensorFaultStsFLC;
    std::uint8_t ADCS8_PA_RPAS_SensorFaultStsSRR;
    std::uint8_t ADCS8_PA_RPAS_SensorFaultStsSRL;
    std::uint8_t ADCS8_PA_FPAS_SensorFaultStsSFR;
    std::uint8_t ADCS8_PA_FPAS_SensorFaultStsSFL;
    std::uint8_t ADCS8_PA_RPAS_SensorFaultStsRRC;
    std::uint8_t ADCS8_PA_RPAS_SensorFaultStsRRM;
    std::uint8_t ADCS8_PA_RPAS_SensorFaultStsRLM;
    std::uint8_t ADCS8_PA_RPAS_SensorFaultStsRLC;
    std::uint8_t ADCS8_Beepreq;
    std::uint8_t ADCS8_PA_WarningType;
    std::uint8_t ADCS8_Mod_Object_MovingDirection;
    std::uint8_t ADCS8_TBA_text;
    std::uint8_t ADCS8_AVM_MODWarning;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::CANFD_Msg194,ADCS5_RPA_slot_ID_9_P0_X,ADCS5_RPA_slot_ID_9_P0_Y,ADCS5_RPA_slot_ID_9_P1_X,ADCS5_RPA_slot_ID_9_P1_Y,ADCS5_RPA_slot_ID_9_P2_X,ADCS5_RPA_slot_ID_9_P2_Y,ADCS5_RPA_slot_ID_9_P3_X,ADCS5_RPA_slot_ID_9_P3_Y,ADCS5_RPA_slot_Angle9,ADCS5_RPA_ParkingSlotDepth9,ADCS5_RPA_ParkingSlotWidth9,ADCS5_RPA_slot_ID_9_Status,ADCS5_RPA_ParkingSlotType9,ADCS5_RPA_ParkingSlotDirection9,ADCS8_PA_warninginfo,ADCS8_PA_FPAS_SensorFaultStsFRC,ADCS8_PA_FPAS_SensorFaultStsFRM,ADCS8_PA_FPAS_SensorFaultStsFLM,ADCS8_PA_FPAS_SensorFaultStsFLC,ADCS8_PA_RPAS_SensorFaultStsSRR,ADCS8_PA_RPAS_SensorFaultStsSRL,ADCS8_PA_FPAS_SensorFaultStsSFR,ADCS8_PA_FPAS_SensorFaultStsSFL,ADCS8_PA_RPAS_SensorFaultStsRRC,ADCS8_PA_RPAS_SensorFaultStsRRM,ADCS8_PA_RPAS_SensorFaultStsRLM,ADCS8_PA_RPAS_SensorFaultStsRLC,ADCS8_Beepreq,ADCS8_PA_WarningType,ADCS8_Mod_Object_MovingDirection,ADCS8_TBA_text,ADCS8_AVM_MODWarning);

#endif // HOZON_NETAOS_IMPL_TYPE_CANFD_MSG194_H_
/* EOF */
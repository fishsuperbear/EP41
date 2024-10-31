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
 * @file impl_type_canfd_msg191.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_CANFD_MSG191_H_
#define HOZON_NETAOS_IMPL_TYPE_CANFD_MSG191_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct CANFD_Msg191 {
    float ADCS6_RPA_slot_ID_4_P0_X;
    float ADCS6_RPA_slot_ID_4_P0_Y;
    float ADCS6_RPA_slot_ID_4_P1_X;
    float ADCS6_RPA_slot_ID_4_P1_Y;
    float ADCS6_RPA_slot_ID_4_P2_X;
    float ADCS6_RPA_slot_ID_4_P2_Y;
    float ADCS6_RPA_slot_ID_4_P3_X;
    float ADCS6_RPA_slot_ID_4_P3_Y;
    float ADCS6_RPA_slot_Angle4;
    float ADCS6_RPA_ParkingSlotDepth4;
    float ADCS6_RPA_ParkingSlotWidth4;
    std::uint8_t ADCS6_RPA_slot_ID_4_Status;
    std::uint8_t ADCS6_RPA_ParkingSlotType4;
    std::uint8_t ADCS6_RPA_ParkingSlotDirection4;
    float ADCS6_RPA_slot_ID_5_P0_X;
    float ADCS6_RPA_slot_ID_5_P0_Y;
    float ADCS6_RPA_slot_ID_5_P1_X;
    float ADCS6_RPA_slot_ID_5_P1_Y;
    float ADCS6_RPA_slot_ID_5_P2_X;
    float ADCS6_RPA_slot_ID_5_P2_Y;
    float ADCS6_RPA_slot_ID_5_P3_X;
    float ADCS6_RPA_slot_ID_5_P3_Y;
    float ADCS6_RPA_slot_Angle5;
    float ADCS6_RPA_ParkingSlotDepth5;
    float ADCS6_RPA_ParkingSlotWidth5;
    std::uint8_t ADCS6_RPA_slot_ID_5_Status;
    std::uint8_t ADCS6_RPA_ParkingSlotType5;
    std::uint8_t ADCS6_RPA_ParkingSlotDirection5;
    float ADCS6_RPA_slot_ID_6_P0_X;
    float ADCS6_RPA_slot_ID_6_P0_Y;
    float ADCS6_RPA_slot_ID_6_P1_X;
    float ADCS6_RPA_slot_ID_6_P1_Y;
    float ADCS6_RPA_slot_ID_6_P2_X;
    float ADCS6_RPA_slot_ID_6_P2_Y;
    float ADCS6_RPA_slot_ID_6_P3_X;
    float ADCS6_RPA_slot_ID_6_P3_Y;
    float ADCS6_RPA_slot_Angle6;
    float ADCS6_RPA_ParkingSlotDepth6;
    float ADCS6_RPA_ParkingSlotWidth6;
    std::uint8_t ADCS6_RPA_slot_ID_6_Status;
    std::uint8_t ADCS6_RPA_ParkingSlotType6;
    std::uint8_t ADCS6_RPA_ParkingSlotDirection6;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::CANFD_Msg191,ADCS6_RPA_slot_ID_4_P0_X,ADCS6_RPA_slot_ID_4_P0_Y,ADCS6_RPA_slot_ID_4_P1_X,ADCS6_RPA_slot_ID_4_P1_Y,ADCS6_RPA_slot_ID_4_P2_X,ADCS6_RPA_slot_ID_4_P2_Y,ADCS6_RPA_slot_ID_4_P3_X,ADCS6_RPA_slot_ID_4_P3_Y,ADCS6_RPA_slot_Angle4,ADCS6_RPA_ParkingSlotDepth4,ADCS6_RPA_ParkingSlotWidth4,ADCS6_RPA_slot_ID_4_Status,ADCS6_RPA_ParkingSlotType4,ADCS6_RPA_ParkingSlotDirection4,ADCS6_RPA_slot_ID_5_P0_X,ADCS6_RPA_slot_ID_5_P0_Y,ADCS6_RPA_slot_ID_5_P1_X,ADCS6_RPA_slot_ID_5_P1_Y,ADCS6_RPA_slot_ID_5_P2_X,ADCS6_RPA_slot_ID_5_P2_Y,ADCS6_RPA_slot_ID_5_P3_X,ADCS6_RPA_slot_ID_5_P3_Y,ADCS6_RPA_slot_Angle5,ADCS6_RPA_ParkingSlotDepth5,ADCS6_RPA_ParkingSlotWidth5,ADCS6_RPA_slot_ID_5_Status,ADCS6_RPA_ParkingSlotType5,ADCS6_RPA_ParkingSlotDirection5,ADCS6_RPA_slot_ID_6_P0_X,ADCS6_RPA_slot_ID_6_P0_Y,ADCS6_RPA_slot_ID_6_P1_X,ADCS6_RPA_slot_ID_6_P1_Y,ADCS6_RPA_slot_ID_6_P2_X,ADCS6_RPA_slot_ID_6_P2_Y,ADCS6_RPA_slot_ID_6_P3_X,ADCS6_RPA_slot_ID_6_P3_Y,ADCS6_RPA_slot_Angle6,ADCS6_RPA_ParkingSlotDepth6,ADCS6_RPA_ParkingSlotWidth6,ADCS6_RPA_slot_ID_6_Status,ADCS6_RPA_ParkingSlotType6,ADCS6_RPA_ParkingSlotDirection6);

#endif // HOZON_NETAOS_IMPL_TYPE_CANFD_MSG191_H_
/* EOF */
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
 * @file impl_type_canfd_msg190.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_CANFD_MSG190_H_
#define HOZON_NETAOS_IMPL_TYPE_CANFD_MSG190_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct CANFD_Msg190 {
    float ADCS5_RPA_slot_ID_1_P0_X;
    float ADCS5_RPA_slot_ID_1_P0_Y;
    float ADCS5_RPA_slot_ID_1_P1_X;
    float ADCS5_RPA_slot_ID_1_P1_Y;
    float ADCS5_RPA_slot_ID_1_P2_X;
    float ADCS5_RPA_slot_ID_1_P2_Y;
    float ADCS5_RPA_slot_ID_1_P3_X;
    float ADCS5_RPA_slot_ID_1_P3_Y;
    float ADCS5_RPA_slot_Angle1;
    float ADCS5_RPA_ParkingSlotDepth1;
    float ADCS5_RPA_ParkingSlotWidth1;
    std::uint8_t ADCS5_RPA_slot_ID_1_Status;
    std::uint8_t ADCS5_RPA_ParkingSlotType1;
    std::uint8_t ADCS5_RPA_ParkingSlotDirection1;
    float ADCS5_RPA_slot_ID_2_P0_X;
    float ADCS5_RPA_slot_ID_2_P0_Y;
    float ADCS5_RPA_slot_ID_2_P1_X;
    float ADCS5_RPA_slot_ID_2_P1_Y;
    float ADCS5_RPA_slot_ID_2_P2_X;
    float ADCS5_RPA_slot_ID_2_P2_Y;
    float ADCS5_RPA_slot_ID_2_P3_X;
    float ADCS5_RPA_slot_ID_2_P3_Y;
    float ADCS5_RPA_slot_Angle2;
    float ADCS5_RPA_ParkingSlotDepth2;
    float ADCS5_RPA_ParkingSlotWidth2;
    std::uint8_t ADCS5_RPA_slot_ID_2_Status;
    std::uint8_t ADCS5_RPA_ParkingSlotType2;
    std::uint8_t ADCS5_RPA_ParkingSlotDirection2;
    float ADCS5_RPA_slot_ID_3_P0_X;
    float ADCS5_RPA_slot_ID_3_P0_Y;
    float ADCS5_RPA_slot_ID_3_P1_X;
    float ADCS5_RPA_slot_ID_3_P1_Y;
    float ADCS5_RPA_slot_ID_3_P2_X;
    float ADCS5_RPA_slot_ID_3_P2_Y;
    float ADCS5_RPA_slot_ID_3_P3_X;
    float ADCS5_RPA_slot_ID_3_P3_Y;
    float ADCS5_RPA_slot_Angle3;
    float ADCS5_RPA_ParkingSlotDepth3;
    float ADCS5_RPA_ParkingSlotWidth3;
    std::uint8_t ADCS5_RPA_slot_ID_3_Status;
    std::uint8_t ADCS5_RPA_ParkingSlotType3;
    std::uint8_t ADCS5_RPA_ParkingSlotDirection3;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::CANFD_Msg190,ADCS5_RPA_slot_ID_1_P0_X,ADCS5_RPA_slot_ID_1_P0_Y,ADCS5_RPA_slot_ID_1_P1_X,ADCS5_RPA_slot_ID_1_P1_Y,ADCS5_RPA_slot_ID_1_P2_X,ADCS5_RPA_slot_ID_1_P2_Y,ADCS5_RPA_slot_ID_1_P3_X,ADCS5_RPA_slot_ID_1_P3_Y,ADCS5_RPA_slot_Angle1,ADCS5_RPA_ParkingSlotDepth1,ADCS5_RPA_ParkingSlotWidth1,ADCS5_RPA_slot_ID_1_Status,ADCS5_RPA_ParkingSlotType1,ADCS5_RPA_ParkingSlotDirection1,ADCS5_RPA_slot_ID_2_P0_X,ADCS5_RPA_slot_ID_2_P0_Y,ADCS5_RPA_slot_ID_2_P1_X,ADCS5_RPA_slot_ID_2_P1_Y,ADCS5_RPA_slot_ID_2_P2_X,ADCS5_RPA_slot_ID_2_P2_Y,ADCS5_RPA_slot_ID_2_P3_X,ADCS5_RPA_slot_ID_2_P3_Y,ADCS5_RPA_slot_Angle2,ADCS5_RPA_ParkingSlotDepth2,ADCS5_RPA_ParkingSlotWidth2,ADCS5_RPA_slot_ID_2_Status,ADCS5_RPA_ParkingSlotType2,ADCS5_RPA_ParkingSlotDirection2,ADCS5_RPA_slot_ID_3_P0_X,ADCS5_RPA_slot_ID_3_P0_Y,ADCS5_RPA_slot_ID_3_P1_X,ADCS5_RPA_slot_ID_3_P1_Y,ADCS5_RPA_slot_ID_3_P2_X,ADCS5_RPA_slot_ID_3_P2_Y,ADCS5_RPA_slot_ID_3_P3_X,ADCS5_RPA_slot_ID_3_P3_Y,ADCS5_RPA_slot_Angle3,ADCS5_RPA_ParkingSlotDepth3,ADCS5_RPA_ParkingSlotWidth3,ADCS5_RPA_slot_ID_3_Status,ADCS5_RPA_ParkingSlotType3,ADCS5_RPA_ParkingSlotDirection3);

#endif // HOZON_NETAOS_IMPL_TYPE_CANFD_MSG190_H_
/* EOF */
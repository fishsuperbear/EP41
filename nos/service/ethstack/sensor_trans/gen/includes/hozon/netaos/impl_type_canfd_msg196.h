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
 * @file impl_type_canfd_msg196.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_CANFD_MSG196_H_
#define HOZON_NETAOS_IMPL_TYPE_CANFD_MSG196_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct CANFD_Msg196 {
    float ADCS5_RPA_slot_ID_10_P0_X;
    float ADCS5_RPA_slot_ID_10_P0_Y;
    float ADCS5_RPA_slot_ID_10_P1_X;
    float ADCS5_RPA_slot_ID_10_P1_Y;
    float ADCS5_RPA_slot_ID_10_P2_X;
    float ADCS5_RPA_slot_ID_10_P2_Y;
    float ADCS5_RPA_slot_ID_10_P3_X;
    float ADCS5_RPA_slot_ID_10_P3_Y;
    float ADCS5_RPA_slot_Angle10;
    float ADCS5_RPA_ParkingSlotDepth10;
    float ADCS5_RPA_ParkingSlotWidth10;
    std::uint8_t ADCS5_RPA_slot_ID_10_Status;
    std::uint8_t ADCS5_RPA_ParkingSlotType10;
    std::uint8_t ADCS5_RPA_ParkingSlotDirection10;
    float ADCS5_RPA_slot_ID_11_P0_X;
    float ADCS5_RPA_slot_ID_11_P0_Y;
    float ADCS5_RPA_slot_ID_11_P1_X;
    float ADCS5_RPA_slot_ID_11_P1_Y;
    float ADCS5_RPA_slot_ID_11_P2_X;
    float ADCS5_RPA_slot_ID_11_P2_Y;
    float ADCS5_RPA_slot_ID_11_P3_X;
    float ADCS5_RPA_slot_ID_11_P3_Y;
    float ADCS5_RPA_slot_Angle11;
    float ADCS5_RPA_ParkingSlotDepth11;
    float ADCS5_RPA_ParkingSlotWidth11;
    std::uint8_t ADCS5_RPA_slot_ID_11_Status;
    std::uint8_t ADCS5_RPA_ParkingSlotType11;
    std::uint8_t ADCS5_RPA_ParkingSlotDirection11;
    float ADCS5_RPA_slot_ID_12_P0_X;
    float ADCS5_RPA_slot_ID_12_P0_Y;
    float ADCS5_RPA_slot_ID_12_P1_X;
    float ADCS5_RPA_slot_ID_12_P1_Y;
    float ADCS5_RPA_slot_ID_12_P2_X;
    float ADCS5_RPA_slot_ID_12_P2_Y;
    float ADCS5_RPA_slot_ID_12_P3_X;
    float ADCS5_RPA_slot_ID_12_P3_Y;
    float ADCS5_RPA_slot_Angle12;
    float ADCS5_RPA_ParkingSlotDepth12;
    float ADCS5_RPA_ParkingSlotWidth12;
    std::uint8_t ADCS5_RPA_slot_ID_12_Status;
    std::uint8_t ADCS5_RPA_ParkingSlotType12;
    std::uint8_t ADCS5_RPA_ParkingSlotDirection12;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::CANFD_Msg196,ADCS5_RPA_slot_ID_10_P0_X,ADCS5_RPA_slot_ID_10_P0_Y,ADCS5_RPA_slot_ID_10_P1_X,ADCS5_RPA_slot_ID_10_P1_Y,ADCS5_RPA_slot_ID_10_P2_X,ADCS5_RPA_slot_ID_10_P2_Y,ADCS5_RPA_slot_ID_10_P3_X,ADCS5_RPA_slot_ID_10_P3_Y,ADCS5_RPA_slot_Angle10,ADCS5_RPA_ParkingSlotDepth10,ADCS5_RPA_ParkingSlotWidth10,ADCS5_RPA_slot_ID_10_Status,ADCS5_RPA_ParkingSlotType10,ADCS5_RPA_ParkingSlotDirection10,ADCS5_RPA_slot_ID_11_P0_X,ADCS5_RPA_slot_ID_11_P0_Y,ADCS5_RPA_slot_ID_11_P1_X,ADCS5_RPA_slot_ID_11_P1_Y,ADCS5_RPA_slot_ID_11_P2_X,ADCS5_RPA_slot_ID_11_P2_Y,ADCS5_RPA_slot_ID_11_P3_X,ADCS5_RPA_slot_ID_11_P3_Y,ADCS5_RPA_slot_Angle11,ADCS5_RPA_ParkingSlotDepth11,ADCS5_RPA_ParkingSlotWidth11,ADCS5_RPA_slot_ID_11_Status,ADCS5_RPA_ParkingSlotType11,ADCS5_RPA_ParkingSlotDirection11,ADCS5_RPA_slot_ID_12_P0_X,ADCS5_RPA_slot_ID_12_P0_Y,ADCS5_RPA_slot_ID_12_P1_X,ADCS5_RPA_slot_ID_12_P1_Y,ADCS5_RPA_slot_ID_12_P2_X,ADCS5_RPA_slot_ID_12_P2_Y,ADCS5_RPA_slot_ID_12_P3_X,ADCS5_RPA_slot_ID_12_P3_Y,ADCS5_RPA_slot_Angle12,ADCS5_RPA_ParkingSlotDepth12,ADCS5_RPA_ParkingSlotWidth12,ADCS5_RPA_slot_ID_12_Status,ADCS5_RPA_ParkingSlotType12,ADCS5_RPA_ParkingSlotDirection12);

#endif // HOZON_NETAOS_IMPL_TYPE_CANFD_MSG196_H_
/* EOF */
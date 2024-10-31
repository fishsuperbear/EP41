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
 * @file impl_type_canfd_msg192.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_CANFD_MSG192_H_
#define HOZON_NETAOS_IMPL_TYPE_CANFD_MSG192_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct CANFD_Msg192 {
    float ADCS7_RPA_slot_ID_7_P0_X;
    float ADCS7_RPA_slot_ID_7_P0_Y;
    float ADCS7_RPA_slot_ID_7_P1_X;
    float ADCS7_RPA_slot_ID_7_P1_Y;
    float ADCS7_RPA_slot_ID_7_P2_X;
    float ADCS7_RPA_slot_ID_7_P2_Y;
    float ADCS7_RPA_slot_ID_7_P3_X;
    float ADCS7_RPA_slot_ID_7_P3_Y;
    float ADCS7_RPA_slot_Angle7;
    float ADCS7_RPA_ParkingSlotDepth7;
    float ADCS7_RPA_ParkingSlotWidth7;
    std::uint8_t ADCS7_RPA_slot_ID_7_Status;
    std::uint8_t ADCS7_RPA_ParkingSlotType7;
    std::uint8_t ADCS7_RPA_ParkingSlotDirection7;
    float ADCS7_RPA_slot_ID_8_P0_X;
    float ADCS7_RPA_slot_ID_8_P0_Y;
    float ADCS7_RPA_slot_ID_8_P1_X;
    float ADCS7_RPA_slot_ID_8_P1_Y;
    float ADCS7_RPA_slot_ID_8_P2_X;
    float ADCS7_RPA_slot_ID_8_P2_Y;
    float ADCS7_RPA_slot_ID_8_P3_X;
    float ADCS7_RPA_slot_ID_8_P3_Y;
    float ADCS7_RPA_slot_Angle8;
    float ADCS7_RPA_ParkingSlotDepth8;
    float ADCS7_RPA_ParkingSlotWidth8;
    std::uint8_t ADCS7_RPA_slot_ID_8_Status;
    std::uint8_t ADCS7_RPA_ParkingSlotType8;
    std::uint8_t ADCS7_RPA_ParkingSlotDirection8;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::CANFD_Msg192,ADCS7_RPA_slot_ID_7_P0_X,ADCS7_RPA_slot_ID_7_P0_Y,ADCS7_RPA_slot_ID_7_P1_X,ADCS7_RPA_slot_ID_7_P1_Y,ADCS7_RPA_slot_ID_7_P2_X,ADCS7_RPA_slot_ID_7_P2_Y,ADCS7_RPA_slot_ID_7_P3_X,ADCS7_RPA_slot_ID_7_P3_Y,ADCS7_RPA_slot_Angle7,ADCS7_RPA_ParkingSlotDepth7,ADCS7_RPA_ParkingSlotWidth7,ADCS7_RPA_slot_ID_7_Status,ADCS7_RPA_ParkingSlotType7,ADCS7_RPA_ParkingSlotDirection7,ADCS7_RPA_slot_ID_8_P0_X,ADCS7_RPA_slot_ID_8_P0_Y,ADCS7_RPA_slot_ID_8_P1_X,ADCS7_RPA_slot_ID_8_P1_Y,ADCS7_RPA_slot_ID_8_P2_X,ADCS7_RPA_slot_ID_8_P2_Y,ADCS7_RPA_slot_ID_8_P3_X,ADCS7_RPA_slot_ID_8_P3_Y,ADCS7_RPA_slot_Angle8,ADCS7_RPA_ParkingSlotDepth8,ADCS7_RPA_ParkingSlotWidth8,ADCS7_RPA_slot_ID_8_Status,ADCS7_RPA_ParkingSlotType8,ADCS7_RPA_ParkingSlotDirection8);

#endif // HOZON_NETAOS_IMPL_TYPE_CANFD_MSG192_H_
/* EOF */
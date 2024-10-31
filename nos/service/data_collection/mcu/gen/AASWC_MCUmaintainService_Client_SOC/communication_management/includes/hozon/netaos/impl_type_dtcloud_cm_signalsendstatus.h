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
 * @file impl_type_dtcloud_cm_signalsendstatus.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_SIGNALSENDSTATUS_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_SIGNALSENDSTATUS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_CM_SignalSendStatus {
    std::uint16_t CM_Send_0x3D8_Cnt;
    std::uint16_t CM_Send_0x136_Cnt;
    std::uint16_t CM_Send_0x265_Cnt;
    std::uint16_t CM_Send_0x8E_Cnt;
    std::uint16_t CM_Send_0xFE_Cnt;
    std::uint16_t CM_Send_0x190_Cnt;
    std::uint16_t CM_Send_0x191_Cnt;
    std::uint16_t CM_Send_0x192_Cnt;
    std::uint16_t CM_Send_0x193_Cnt;
    std::uint16_t CM_Send_0x210_Cnt;
    std::uint16_t CM_Send_0x194_Cnt;
    std::uint16_t CM_Send_0x8F_Cnt;
    std::uint16_t CM_Send_0x255_Cnt;
    std::uint16_t CM_Send_0x301_Cnt;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_CM_SignalSendStatus,CM_Send_0x3D8_Cnt,CM_Send_0x136_Cnt,CM_Send_0x265_Cnt,CM_Send_0x8E_Cnt,CM_Send_0xFE_Cnt,CM_Send_0x190_Cnt,CM_Send_0x191_Cnt,CM_Send_0x192_Cnt,CM_Send_0x193_Cnt,CM_Send_0x210_Cnt,CM_Send_0x194_Cnt,CM_Send_0x8F_Cnt,CM_Send_0x255_Cnt,CM_Send_0x301_Cnt);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_CM_SIGNALSENDSTATUS_H_
/* EOF */
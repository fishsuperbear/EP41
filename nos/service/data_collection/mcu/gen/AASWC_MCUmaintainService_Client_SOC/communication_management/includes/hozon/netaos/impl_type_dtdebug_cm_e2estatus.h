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
 * @file impl_type_dtdebug_cm_e2estatus.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_E2ESTATUS_H_
#define HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_E2ESTATUS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtDebug_CM_E2EStatus {
    std::uint8_t CM_E2E_FD3_0x137;
    std::uint8_t CM_E2E_FD3_0x138;
    std::uint8_t CM_E2E_FD3_0x139;
    std::uint8_t CM_E2E_FD3_0x13B;
    std::uint8_t CM_E2E_FD3_0x13C;
    std::uint8_t CM_E2E_FD3_0x13D;
    std::uint8_t CM_E2E_FD3_0x13E;
    std::uint8_t CM_E2E_FD3_0xC4;
    std::uint8_t CM_E2E_FD3_0x110;
    std::uint8_t CM_E2E_FD3_0x114;
    std::uint8_t CM_E2E_FD3_0xF3;
    std::uint8_t CM_E2E_FD3_0xB1;
    std::uint8_t CM_E2E_FD3_0xB2;
    std::uint8_t CM_E2E_FD3_0xAB;
    std::uint8_t CM_E2E_FD3_0xC0;
    std::uint8_t CM_E2E_FD3_0xC5;
    std::uint8_t CM_E2E_FD3_0xC7;
    std::uint8_t CM_E2E_FD3_0xE5;
    std::uint8_t CM_E2E_FD3_0x121;
    std::uint8_t CM_E2E_FD3_0x129;
    std::uint8_t CM_E2E_FD3_0x108;
    std::uint8_t CM_E2E_FD3_0x1B6;
    std::uint8_t CM_E2E_FD3_0xE3;
    std::uint8_t CM_E2E_FD3_0x12D;
    std::uint8_t CM_E2E_FD6_0x110;
    std::uint8_t CM_E2E_FD8_0x137;
    std::uint8_t CM_E2E_FD8_0x138;
    std::uint8_t CM_E2E_FD8_0x139;
    std::uint8_t CM_E2E_FD8_0x13B;
    std::uint8_t CM_E2E_FD8_0x13C;
    std::uint8_t CM_E2E_FD8_0x13D;
    std::uint8_t CM_E2E_FD8_0x13E;
    std::uint8_t CM_E2E_FD8_0x2FE;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtDebug_CM_E2EStatus,CM_E2E_FD3_0x137,CM_E2E_FD3_0x138,CM_E2E_FD3_0x139,CM_E2E_FD3_0x13B,CM_E2E_FD3_0x13C,CM_E2E_FD3_0x13D,CM_E2E_FD3_0x13E,CM_E2E_FD3_0xC4,CM_E2E_FD3_0x110,CM_E2E_FD3_0x114,CM_E2E_FD3_0xF3,CM_E2E_FD3_0xB1,CM_E2E_FD3_0xB2,CM_E2E_FD3_0xAB,CM_E2E_FD3_0xC0,CM_E2E_FD3_0xC5,CM_E2E_FD3_0xC7,CM_E2E_FD3_0xE5,CM_E2E_FD3_0x121,CM_E2E_FD3_0x129,CM_E2E_FD3_0x108,CM_E2E_FD3_0x1B6,CM_E2E_FD3_0xE3,CM_E2E_FD3_0x12D,CM_E2E_FD6_0x110,CM_E2E_FD8_0x137,CM_E2E_FD8_0x138,CM_E2E_FD8_0x139,CM_E2E_FD8_0x13B,CM_E2E_FD8_0x13C,CM_E2E_FD8_0x13D,CM_E2E_FD8_0x13E,CM_E2E_FD8_0x2FE);

#endif // HOZON_NETAOS_IMPL_TYPE_DTDEBUG_CM_E2ESTATUS_H_
/* EOF */
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
 * @file impl_type_dtcloud_hafmsgsoctomcualivefltinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFMSGSOCTOMCUALIVEFLTINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFMSGSOCTOMCUALIVEFLTINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_HafmsgSocToMcuAliveFltInfo {
    std::uint8_t MsgAliveFlt_0x100;
    std::uint8_t MsgAliveFlt_0x112;
    std::uint8_t MsgAliveFlt_0x1A2;
    std::uint8_t MsgAliveFlt_0x0A2;
    std::uint8_t MsgAliveFlt_0x0E3;
    std::uint8_t MsgAliveFlt_0x0E5;
    std::uint8_t MsgAliveFlt_0x200;
    std::uint8_t MsgAliveFlt_0x201;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_HafmsgSocToMcuAliveFltInfo,MsgAliveFlt_0x100,MsgAliveFlt_0x112,MsgAliveFlt_0x1A2,MsgAliveFlt_0x0A2,MsgAliveFlt_0x0E3,MsgAliveFlt_0x0E5,MsgAliveFlt_0x200,MsgAliveFlt_0x201);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFMSGSOCTOMCUALIVEFLTINFO_H_
/* EOF */
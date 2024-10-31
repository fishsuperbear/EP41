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
 * @file impl_type_algmcutoegoframe.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGMCUTOEGOFRAME_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGMCUTOEGOFRAME_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_algmcuegoavpmsg.h"
#include "hozon/netaos/impl_type_algmcuegonnpmsg.h"
#include "hozon/netaos/impl_type_hafheader.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgMcuToEgoFrame {
    ::hozon::netaos::HafHeader header;
    ::hozon::netaos::AlgMcuEgoNNPMsg msg_mcu_nnp;
    ::hozon::netaos::AlgMcuEgoAVPMsg msg_mcu_avp;
    std::uint8_t ta_pilot_mode;
    std::uint32_t FCT2SOC_TBD_u32_01;
    std::uint32_t FCT2SOC_TBD_u32_02;
    std::uint32_t FCT2SOC_TBD_u32_03;
    std::uint32_t FCT2SOC_TBD_u32_04;
    std::uint32_t FCT2SOC_TBD_u32_05;
    std::uint8_t drive_mode;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgMcuToEgoFrame,header,msg_mcu_nnp,msg_mcu_avp,ta_pilot_mode,FCT2SOC_TBD_u32_01,FCT2SOC_TBD_u32_02,FCT2SOC_TBD_u32_03,FCT2SOC_TBD_u32_04,FCT2SOC_TBD_u32_05,drive_mode);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGMCUTOEGOFRAME_H_
/* EOF */
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
 * @file impl_type_algegotomcuframe.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGEGOTOMCUFRAME_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGEGOTOMCUFRAME_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_algegomcuavpmsg.h"
#include "hozon/netaos/impl_type_algegomcunnpmsg.h"
#include "hozon/netaos/impl_type_hafheader.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgEgoToMcuFrame {
    ::hozon::netaos::HafHeader header;
    ::hozon::netaos::AlgEgoMcuNNPMsg msg_ego_nnp;
    ::hozon::netaos::AlgEgoMcuAVPMsg msg_ego_avp;
    std::uint32_t SOC2FCT_TBD_u32_01;
    std::uint32_t SOC2FCT_TBD_u32_02;
    std::uint32_t SOC2FCT_TBD_u32_03;
    std::uint32_t SOC2FCT_TBD_u32_04;
    std::uint32_t SOC2FCT_TBD_u32_05;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgEgoToMcuFrame,header,msg_ego_nnp,msg_ego_avp,SOC2FCT_TBD_u32_01,SOC2FCT_TBD_u32_02,SOC2FCT_TBD_u32_03,SOC2FCT_TBD_u32_04,SOC2FCT_TBD_u32_05);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGEGOTOMCUFRAME_H_
/* EOF */
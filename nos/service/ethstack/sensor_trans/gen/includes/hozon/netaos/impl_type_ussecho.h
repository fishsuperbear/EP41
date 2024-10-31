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
 * @file impl_type_ussecho.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_USSECHO_H_
#define HOZON_NETAOS_IMPL_TYPE_USSECHO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_ussechodistance_a.h"
#include "hozon/netaos/impl_type_ussechopeak_a.h"
#include "hozon/netaos/impl_type_ussechowidth_a.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct UssEcho {
    std::uint8_t echo_num;
    ::hozon::netaos::UssEchoDistance_A distance;
    ::hozon::netaos::UssEchoWidth_A width;
    ::hozon::netaos::UssEchoPeak_A peak;
    std::uint8_t status_error;
    std::uint8_t status_work;
    std::uint16_t wTxSns_Ringtime;
    std::uint8_t counter;
    std::uint64_t system_time;
    std::uint16_t ReservedA;
    std::uint16_t ReservedB;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::UssEcho,echo_num,distance,width,peak,status_error,status_work,wTxSns_Ringtime,counter,system_time,ReservedA,ReservedB);

#endif // HOZON_NETAOS_IMPL_TYPE_USSECHO_H_
/* EOF */
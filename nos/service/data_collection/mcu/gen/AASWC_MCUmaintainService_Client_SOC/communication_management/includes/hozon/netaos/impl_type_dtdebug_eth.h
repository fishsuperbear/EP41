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
 * @file impl_type_dtdebug_eth.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTDEBUG_ETH_H_
#define HOZON_NETAOS_IMPL_TYPE_DTDEBUG_ETH_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtdebug_eth_array_1010.h"
#include "hozon/netaos/impl_type_dtdebug_eth_array_1011.h"
#include "hozon/netaos/impl_type_dtdebug_eth_array_1012.h"
#include "hozon/netaos/impl_type_dtdebug_eth_array_1013.h"
#include "hozon/netaos/impl_type_dtdebug_eth_array_1014.h"
#include "hozon/netaos/impl_type_dtdebug_eth_array_1015.h"
#include "hozon/netaos/impl_type_dtdebug_eth_array_1016.h"
#include "hozon/netaos/impl_type_dtdebug_hafglobaltime.h"
namespace hozon {
namespace netaos {
struct DtDebug_ETH {
    ::hozon::netaos::DtDebug_ETH_Array_1010 Sd_ServerServicehz;
    ::hozon::netaos::DtDebug_ETH_Array_1011 Sd_ClientServicehz;
    ::hozon::netaos::DtDebug_ETH_Array_1012 Sd_ServerServiceRuntimehz;
    ::hozon::netaos::DtDebug_ETH_Array_1013 Sd_ClientServiceRuntimehz;
    ::hozon::netaos::DtDebug_ETH_Array_1014 SomeIpTp_RxDatahz;
    ::hozon::netaos::DtDebug_ETH_Array_1015 SomeIpTp_TxDatahz;
    ::hozon::netaos::DtDebug_ETH_Array_1016 counterseqhz;
    ::hozon::netaos::DtDebug_HafGlobalTime HafGlobalTime;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtDebug_ETH,Sd_ServerServicehz,Sd_ClientServicehz,Sd_ServerServiceRuntimehz,Sd_ClientServiceRuntimehz,SomeIpTp_RxDatahz,SomeIpTp_TxDatahz,counterseqhz,HafGlobalTime);

#endif // HOZON_NETAOS_IMPL_TYPE_DTDEBUG_ETH_H_
/* EOF */
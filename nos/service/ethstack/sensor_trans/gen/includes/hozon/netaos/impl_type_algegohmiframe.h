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
 * @file impl_type_algegohmiframe.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGEGOHMIFRAME_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGEGOHMIFRAME_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_algegoparkhmiinfo.h"
#include "hozon/netaos/impl_type_algegotsrihbchmiinfo.h"
#include "hozon/netaos/impl_type_algegowarninginfo.h"
#include "hozon/netaos/impl_type_hafheader.h"
namespace hozon {
namespace netaos {
struct AlgEgoHmiFrame {
    ::hozon::netaos::HafHeader header;
    bool isValid;
    ::hozon::netaos::AlgEgoWarningInfo warnning_info;
    ::hozon::netaos::AlgEgoParkHmiInfo park_hmi_info;
    ::hozon::netaos::AlgEgoTsrIhbcHmiInfo tsr_ihbc_info;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgEgoHmiFrame,header,isValid,warnning_info,park_hmi_info,tsr_ihbc_info);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGEGOHMIFRAME_H_
/* EOF */
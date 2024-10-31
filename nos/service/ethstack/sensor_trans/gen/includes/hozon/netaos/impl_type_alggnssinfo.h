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
 * @file impl_type_alggnssinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGGNSSINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGGNSSINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_alggnssheadinginfo.h"
#include "hozon/netaos/impl_type_alggnssposinfo.h"
#include "hozon/netaos/impl_type_alggnssvelinfo.h"
#include "hozon/netaos/impl_type_hafheader.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgGnssInfo {
    ::hozon::netaos::HafHeader header;
    std::uint32_t gpsWeek;
    double gpsSec;
    ::hozon::netaos::AlgGNSSPosInfo gnss_pos;
    ::hozon::netaos::AlgGNSSVelInfo gnss_vel;
    ::hozon::netaos::AlgGNSSHeadingInfo gnss_heading;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgGnssInfo,header,gpsWeek,gpsSec,gnss_pos,gnss_vel,gnss_heading);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGGNSSINFO_H_
/* EOF */
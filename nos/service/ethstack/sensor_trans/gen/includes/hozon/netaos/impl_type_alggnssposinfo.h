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
 * @file impl_type_alggnssposinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGGNSSPOSINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGGNSSPOSINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgGNSSPosInfo {
    std::uint8_t posType;
    double latitude;
    double longitude;
    float undulation;
    float altitude;
    float latStd;
    float lonStd;
    float hgtStd;
    std::uint8_t svs;
    std::uint8_t solnSVs;
    std::uint8_t diffAge;
    float hdop;
    float vdop;
    float pdop;
    float gdop;
    float tdop;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgGNSSPosInfo,posType,latitude,longitude,undulation,altitude,latStd,lonStd,hgtStd,svs,solnSVs,diffAge,hdop,vdop,pdop,gdop,tdop);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGGNSSPOSINFO_H_
/* EOF */
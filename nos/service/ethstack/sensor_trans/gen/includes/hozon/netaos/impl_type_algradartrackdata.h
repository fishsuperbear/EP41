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
 * @file impl_type_algradartrackdata.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGRADARTRACKDATA_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGRADARTRACKDATA_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_algradarmodedatainfo.h"
#include "hozon/netaos/impl_type_point3d_32.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgRadarTrackData {
    std::uint32_t id;
    ::hozon::netaos::AlgRadarModeDataInfo position;
    ::hozon::netaos::AlgRadarModeDataInfo velocity;
    ::hozon::netaos::AlgRadarModeDataInfo acceleration;
    double rcs;
    double snr;
    double existProbability;
    std::uint8_t movProperty;
    std::uint8_t trackType;
    std::uint16_t trackAge;
    std::uint8_t objObstacleProb;
    std::uint8_t measState;
    ::hozon::netaos::Point3D_32 sizeLWH;
    double orientAgl;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgRadarTrackData,id,position,velocity,acceleration,rcs,snr,existProbability,movProperty,trackType,trackAge,objObstacleProb,measState,sizeLWH,orientAgl);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGRADARTRACKDATA_H_
/* EOF */
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
 * @file impl_type_algfrontradartrackarrayframe.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGFRONTRADARTRACKARRAYFRAME_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGFRONTRADARTRACKARRAYFRAME_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_algfrontradar_tracklist_a.h"
#include "hozon/netaos/impl_type_hafheader.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgFrontRadarTrackArrayFrame {
    ::hozon::netaos::HafHeader header;
    std::uint8_t sensorID;
    std::uint8_t radarState;
    ::hozon::netaos::AlgFrontRadar_trackList_A trackList;
    bool isValid;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgFrontRadarTrackArrayFrame,header,sensorID,radarState,trackList,isValid);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGFRONTRADARTRACKARRAYFRAME_H_
/* EOF */
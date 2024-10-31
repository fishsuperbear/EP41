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
 * @file impl_type_hafegotrajectory.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_HAFEGOTRAJECTORY_H_
#define HOZON_NETAOS_IMPL_TYPE_HAFEGOTRAJECTORY_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_hafheader_traj.h"
#include "hozon/netaos/impl_type_haftrajectorypoint_a.h"
#include "hozon/netaos/impl_type_reserve_a.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct HafEgoTrajectory {
    ::hozon::netaos::HafHeader_traj header;
    std::uint32_t locSeq;
    float trajectoryLength;
    float trajectoryPeriod;
    float proj_heading_offset;
    double trajectoryPoint_reference_x;
    double trajectoryPoint_reference_y;
    ::hozon::netaos::HafTrajectoryPoint_A trajectoryPoints;
    std::uint8_t trajectoryValidPointsSize;
    std::uint8_t isEstop;
    bool isReplanning;
    std::uint8_t gear;
    std::uint8_t trajectoryType;
    std::uint8_t driviningMode;
    std::uint8_t functionMode;
    std::uint8_t utmzoneID;
    ::hozon::netaos::reserve_A reserve;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::HafEgoTrajectory,header,locSeq,trajectoryLength,trajectoryPeriod,proj_heading_offset,trajectoryPoint_reference_x,trajectoryPoint_reference_y,trajectoryPoints,trajectoryValidPointsSize,isEstop,isReplanning,gear,trajectoryType,driviningMode,functionMode,utmzoneID,reserve);

#endif // HOZON_NETAOS_IMPL_TYPE_HAFEGOTRAJECTORY_H_
/* EOF */
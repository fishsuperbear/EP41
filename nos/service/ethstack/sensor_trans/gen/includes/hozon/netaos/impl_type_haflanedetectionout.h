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
 * @file impl_type_haflanedetectionout.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_HAFLANEDETECTIONOUT_H_
#define HOZON_NETAOS_IMPL_TYPE_HAFLANEDETECTIONOUT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_haflanelinefit.h"
#include "hozon/netaos/impl_type_haftime.h"
#include "hozon/netaos/impl_type_point3d_32.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct HafLaneDetectionOut {
    std::int32_t lanelineSeq;
    float geometryConfidence;
    std::uint8_t cls;
    float typeConfidence;
    std::uint8_t color;
    float colorConfidence;
    float laneLineWidth;
    ::hozon::netaos::Point3D_32 keyPointVRF;
    ::hozon::netaos::HafTime timeCreation;
    ::hozon::netaos::HafLanelineFit laneFit;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::HafLaneDetectionOut,lanelineSeq,geometryConfidence,cls,typeConfidence,color,colorConfidence,laneLineWidth,keyPointVRF,timeCreation,laneFit);

#endif // HOZON_NETAOS_IMPL_TYPE_HAFLANEDETECTIONOUT_H_
/* EOF */
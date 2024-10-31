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
 * @file impl_type_hafposewithcovariance.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_HAFPOSEWITHCOVARIANCE_H_
#define HOZON_NETAOS_IMPL_TYPE_HAFPOSEWITHCOVARIANCE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_hafpose.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct HafPoseWithCovariance {
    ::hozon::netaos::HafPose poseWGS;
    ::hozon::netaos::HafPose poseLOCAL;
    ::hozon::netaos::HafPose poseGCJ02;
    ::hozon::netaos::HafPose poseUTM01;
    ::hozon::netaos::HafPose poseUTM02;
    std::uint16_t utmZoneID01;
    std::uint16_t utmZoneID02;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::HafPoseWithCovariance,poseWGS,poseLOCAL,poseGCJ02,poseUTM01,poseUTM02,utmZoneID01,utmZoneID02);

#endif // HOZON_NETAOS_IMPL_TYPE_HAFPOSEWITHCOVARIANCE_H_
/* EOF */
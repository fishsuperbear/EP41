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
 * @file impl_type_alginsinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGINSINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGINSINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_point3d_64.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgInsInfo {
    double latitude;
    double longitude;
    double altitude;
    ::hozon::netaos::Point3D_64 attitude;
    ::hozon::netaos::Point3D_64 linearVelocity;
    ::hozon::netaos::Point3D_64 augularVelocity;
    ::hozon::netaos::Point3D_64 linearAcceleration;
    float heading;
    ::hozon::netaos::Point3D_64 mountingError;
    ::hozon::netaos::Point3D_64 sdPosition;
    ::hozon::netaos::Point3D_64 sdAttitude;
    ::hozon::netaos::Point3D_64 sdVelocity;
    std::uint16_t sysStatus;
    std::uint16_t gpsStatus;
    std::uint16_t sensorUsed;
    float wheelVelocity;
    float odoSF;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgInsInfo,latitude,longitude,altitude,attitude,linearVelocity,augularVelocity,linearAcceleration,heading,mountingError,sdPosition,sdAttitude,sdVelocity,sysStatus,gpsStatus,sensorUsed,wheelVelocity,odoSF);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGINSINFO_H_
/* EOF */
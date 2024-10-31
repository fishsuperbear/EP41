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
 * @file impl_type_algimu.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGIMU_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGIMU_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_imupose.h"
#include "hozon/netaos/impl_type_point3d_64.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgIMU {
    ::hozon::netaos::Point3D_64 angularVelocity;
    ::hozon::netaos::Point3D_64 linearAcceleration;
    ::hozon::netaos::Point3D_64 imuVBAngularVelocity;
    ::hozon::netaos::Point3D_64 imuVBLinearAcceleration;
    std::uint16_t imuStatus;
    float temperature;
    ::hozon::netaos::Point3D_64 gyroOffset;
    ::hozon::netaos::Point3D_64 accelOffset;
    ::hozon::netaos::Point3D_64 ins2antoffset;
    ::hozon::netaos::ImuPose imu2bodyosffet;
    float imuyaw;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgIMU,angularVelocity,linearAcceleration,imuVBAngularVelocity,imuVBLinearAcceleration,imuStatus,temperature,gyroOffset,accelOffset,ins2antoffset,imu2bodyosffet,imuyaw);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGIMU_H_
/* EOF */
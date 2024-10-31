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
 * @file impl_type_haffusionout.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_HAFFUSIONOUT_H_
#define HOZON_NETAOS_IMPL_TYPE_HAFFUSIONOUT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_hafrect3d.h"
#include "hozon/netaos/impl_type_haftime.h"
#include "hozon/netaos/impl_type_point2d.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct HafFusionOut {
    std::uint8_t ObjectID;
    std::uint8_t Type;
    std::uint32_t DetectSensor_Current;
    std::uint32_t DetectSensor_History;
    std::uint8_t MaintenanceStatus;
    std::uint8_t TypeConfidence;
    std::uint8_t ExistenceProbability;
    ::hozon::netaos::HafRect3D RectInfo;
    ::hozon::netaos::Point2D VelocityAbs;
    ::hozon::netaos::Point2D AccelerationAbs;
    ::hozon::netaos::HafTime TimeCreation;
    ::hozon::netaos::HafTime LastUpdatedTime;
    std::uint8_t MotionPattern;
    std::uint8_t MotionPatternHistory;
    std::uint8_t BrakeLightSt;
    std::uint8_t TurnLightSt;
    std::uint8_t NearSide;
    std::uint32_t Age;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::HafFusionOut,ObjectID,Type,DetectSensor_Current,DetectSensor_History,MaintenanceStatus,TypeConfidence,ExistenceProbability,RectInfo,VelocityAbs,AccelerationAbs,TimeCreation,LastUpdatedTime,MotionPattern,MotionPatternHistory,BrakeLightSt,TurnLightSt,NearSide,Age);

#endif // HOZON_NETAOS_IMPL_TYPE_HAFFUSIONOUT_H_
/* EOF */
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
 * @file impl_type_haftrajectorypoint.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_HAFTRAJECTORYPOINT_H_
#define HOZON_NETAOS_IMPL_TYPE_HAFTRAJECTORYPOINT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
namespace hozon {
namespace netaos {
struct HafTrajectoryPoint {
    float timeRelative;
    float x;
    float y;
    float z;
    float theta;
    float curvature;
    float s;
    float speed;
    float acc;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::HafTrajectoryPoint,timeRelative,x,y,z,theta,curvature,s,speed,acc);

#endif // HOZON_NETAOS_IMPL_TYPE_HAFTRAJECTORYPOINT_H_
/* EOF */
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
 * @file impl_type_mbdposcalcdebug.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_MBDPOSCALCDEBUG_H_
#define HOZON_NETAOS_IMPL_TYPE_MBDPOSCALCDEBUG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
namespace hozon {
namespace netaos {
struct MbdPosCalcDebug {
    double posecalc_posedata_timestamp;
    bool posecalc_inputdata_valid;
    bool posecalc_posedata_timecheck;
    bool posecalc_enable;
    float posecalc_timedelay;
    double posedata_world_pos_x;
    double posedata_world_pos_y;
    float posedata_vrf_vel_x;
    float posedata_vrf_vel_y;
    float posedata_vrf_acc_x;
    float posedata_vrf_acc_y;
    float posedata_world_heading;
    float posedata_vrf_yawrate;
    double posecalc_world_pos_x;
    double posecalc_world_pos_y;
    float posecalc_vrf_vel_x;
    float posecalc_vrf_vel_y;
    float posecalc_vrf_acc_x;
    float posecalc_vrf_acc_y;
    float posecalc_world_vel_x;
    float posecalc_world_vel_y;
    float posecalc_world_acc_x;
    float posecalc_world_acc_y;
    float posecalc_world_heading;
    float posecalc_world_pitch;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::MbdPosCalcDebug,posecalc_posedata_timestamp,posecalc_inputdata_valid,posecalc_posedata_timecheck,posecalc_enable,posecalc_timedelay,posedata_world_pos_x,posedata_world_pos_y,posedata_vrf_vel_x,posedata_vrf_vel_y,posedata_vrf_acc_x,posedata_vrf_acc_y,posedata_world_heading,posedata_vrf_yawrate,posecalc_world_pos_x,posecalc_world_pos_y,posecalc_vrf_vel_x,posecalc_vrf_vel_y,posecalc_vrf_acc_x,posecalc_vrf_acc_y,posecalc_world_vel_x,posecalc_world_vel_y,posecalc_world_acc_x,posecalc_world_acc_y,posecalc_world_heading,posecalc_world_pitch);

#endif // HOZON_NETAOS_IMPL_TYPE_MBDPOSCALCDEBUG_H_
/* EOF */
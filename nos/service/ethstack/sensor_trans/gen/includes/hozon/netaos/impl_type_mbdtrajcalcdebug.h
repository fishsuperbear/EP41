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
 * @file impl_type_mbdtrajcalcdebug.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_MBDTRAJCALCDEBUG_H_
#define HOZON_NETAOS_IMPL_TYPE_MBDTRAJCALCDEBUG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct MbdTrajCalcDebug {
    bool trajCalc_trajdata_replaning_flag;
    std::uint32_t trajcalc_trajdata_estop;
    std::uint32_t trajcalc_trajdata_gearcmd;
    std::uint32_t trajcalc_inputdata_valid;
    double trajcalc_trajdata_timestamp;
    double trajcalc_globaltime_timestamp;
    bool trajcalc_trajdata_pointtime_check;
    bool trajcalc_trajdata_timecheck;
    std::uint32_t trajcalc_enable;
    std::uint32_t trajcalc_lon_startpoint_index;
    float trajcalc_lon_linear_ratio;
    float trajcalc_lon_poserrcmd;
    float trajcalc_lon_headingcmd;
    float trajcalc_lon_velcmd;
    float trajcalc_lon_acc_cmd;
    float trajcalc_lon_curvcmd;
    std::uint32_t trajcalc_lonpre_startpoint_index;
    float trajcalc_lonpre_linear_ratio;
    float trajcalc_lonpre_poserrcmd;
    float trajcalc_lonpre_headingcmd;
    float trajcalc_lonpre_velcmd;
    float trajcalc_lonpre_acc_cmd;
    float trajcalc_lonpre_curvrmd;
    double trajcalc_posedata_posex;
    double trajcalc_posedata_posey;
    std::uint32_t trajcalc_lat_startpoint_index;
    float trajcalc_lat_linear_ratio;
    float trajcalc_lat_match_pointx;
    float trajcalc_lat_match_pointy;
    float trajcalc_lat_poserrcmd;
    float trajcalc_lat_headingcmd;
    float trajcalc_lat_velcmd;
    float trajcalc_lat_acc_cmd;
    float trajcalc_lat_curvcmd;
    double trajcalc_posedata_preposex;
    double trajcalc_posedata_preposey;
    std::uint32_t trajcalc_latpre_startpoint_index;
    float trajcalc_latpre_linear_ratio;
    float trajcalc_latpre_match_pointx;
    float trajcalc_latpre_match_pointy;
    float trajcalc_latpre_poserrcmd;
    float trajcalc_latpre_headingcmd;
    float trajcalc_latpre_velcmd;
    float trajcalc_latpre_acc_cmd;
    float trajcalc_latpre_curvcmd;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::MbdTrajCalcDebug,trajCalc_trajdata_replaning_flag,trajcalc_trajdata_estop,trajcalc_trajdata_gearcmd,trajcalc_inputdata_valid,trajcalc_trajdata_timestamp,trajcalc_globaltime_timestamp,trajcalc_trajdata_pointtime_check,trajcalc_trajdata_timecheck,trajcalc_enable,trajcalc_lon_startpoint_index,trajcalc_lon_linear_ratio,trajcalc_lon_poserrcmd,trajcalc_lon_headingcmd,trajcalc_lon_velcmd,trajcalc_lon_acc_cmd,trajcalc_lon_curvcmd,trajcalc_lonpre_startpoint_index,trajcalc_lonpre_linear_ratio,trajcalc_lonpre_poserrcmd,trajcalc_lonpre_headingcmd,trajcalc_lonpre_velcmd,trajcalc_lonpre_acc_cmd,trajcalc_lonpre_curvrmd,trajcalc_posedata_posex,trajcalc_posedata_posey,trajcalc_lat_startpoint_index,trajcalc_lat_linear_ratio,trajcalc_lat_match_pointx,trajcalc_lat_match_pointy,trajcalc_lat_poserrcmd,trajcalc_lat_headingcmd,trajcalc_lat_velcmd,trajcalc_lat_acc_cmd,trajcalc_lat_curvcmd,trajcalc_posedata_preposex,trajcalc_posedata_preposey,trajcalc_latpre_startpoint_index,trajcalc_latpre_linear_ratio,trajcalc_latpre_match_pointx,trajcalc_latpre_match_pointy,trajcalc_latpre_poserrcmd,trajcalc_latpre_headingcmd,trajcalc_latpre_velcmd,trajcalc_latpre_acc_cmd,trajcalc_latpre_curvcmd);

#endif // HOZON_NETAOS_IMPL_TYPE_MBDTRAJCALCDEBUG_H_
/* EOF */
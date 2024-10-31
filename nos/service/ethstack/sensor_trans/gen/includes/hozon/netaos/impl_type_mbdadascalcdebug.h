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
 * @file impl_type_mbdadascalcdebug.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_MBDADASCALCDEBUG_H_
#define HOZON_NETAOS_IMPL_TYPE_MBDADASCALCDEBUG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct MbdADASCalcDebug {
    std::uint8_t adascalc_is_lccactive;
    float adascalc_trajparam_a0;
    float adascalc_trajparam_a1;
    float adascalc_trajparam_a2;
    float adascalc_trajparam_a3;
    std::uint8_t adascalc_trp_replanlevel;
    std::uint8_t adascalc_hostlindchgtoleft_bl;
    std::uint8_t adascalc_hostlindchgtorght_bl;
    std::uint8_t adascalc_accsystemstate;
    std::uint8_t adascalc_accstopreq;
    float adascalc_deltavelocity;
    float adascalc_deltadist;
    float adascalc_drvrseltrgtspd_sg;
    std::uint8_t adascalc_acc_smrsts;
    std::uint8_t adascalc_enable;
    bool adascalc_replanningflag;
    std::uint32_t adascalc_gearcmd;
    std::uint8_t adascalc_estop;
    float cal_adascalc_headdingoffset_rad;
    float adascalc_lat_poserrcmd;
    float adascalc_lat_headingcmd;
    float adascalc_lat_velcmd;
    float adascalc_latpre_curvcmd;
    float adascalc_lon_poserrcmd;
    float adascalc_lon_velcmd;
    float adascalc_a_acctrajcmd;
    std::uint8_t adascalc_is_longtraj_replan;
    float adascalc_m_strajerror;
    float adascalc_trajparamlong_a0;
    float adascalc_trajparamlong_a1;
    float adascalc_trajparamlong_a2;
    float adascalc_trajparamlong_a3;
    float adascalc_trajparamlong_a4;
    float adascalc_trajparamlong_a5;
    float adascalc_v_spdtrajcmd;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::MbdADASCalcDebug,adascalc_is_lccactive,adascalc_trajparam_a0,adascalc_trajparam_a1,adascalc_trajparam_a2,adascalc_trajparam_a3,adascalc_trp_replanlevel,adascalc_hostlindchgtoleft_bl,adascalc_hostlindchgtorght_bl,adascalc_accsystemstate,adascalc_accstopreq,adascalc_deltavelocity,adascalc_deltadist,adascalc_drvrseltrgtspd_sg,adascalc_acc_smrsts,adascalc_enable,adascalc_replanningflag,adascalc_gearcmd,adascalc_estop,cal_adascalc_headdingoffset_rad,adascalc_lat_poserrcmd,adascalc_lat_headingcmd,adascalc_lat_velcmd,adascalc_latpre_curvcmd,adascalc_lon_poserrcmd,adascalc_lon_velcmd,adascalc_a_acctrajcmd,adascalc_is_longtraj_replan,adascalc_m_strajerror,adascalc_trajparamlong_a0,adascalc_trajparamlong_a1,adascalc_trajparamlong_a2,adascalc_trajparamlong_a3,adascalc_trajparamlong_a4,adascalc_trajparamlong_a5,adascalc_v_spdtrajcmd);

#endif // HOZON_NETAOS_IMPL_TYPE_MBDADASCALCDEBUG_H_
/* EOF */
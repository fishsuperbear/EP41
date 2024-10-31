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
 * @file impl_type_hafmbddebug.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_HAFMBDDEBUG_H_
#define HOZON_NETAOS_IMPL_TYPE_HAFMBDDEBUG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_mbdadascalcdebug.h"
#include "hozon/netaos/impl_type_mbdctrldecdebug.h"
#include "hozon/netaos/impl_type_mbdctrloutputdebug.h"
#include "hozon/netaos/impl_type_mbdlatctrldebug.h"
#include "hozon/netaos/impl_type_mbdlonctrldebug.h"
#include "hozon/netaos/impl_type_mbdposcalcdebug.h"
#include "hozon/netaos/impl_type_mbdtrajcalcdebug.h"
#include "hozon/netaos/impl_type_versionnumber_a.h"
namespace hozon {
namespace netaos {
struct HafMbdDebug {
    ::hozon::netaos::MbdPosCalcDebug pos_calc_debug;
    ::hozon::netaos::MbdTrajCalcDebug traj_calc_debug;
    ::hozon::netaos::MbdADASCalcDebug adas_calc_debug;
    ::hozon::netaos::MbdCtrlDecDebug ctrl_dec_debug;
    ::hozon::netaos::MbdLonCtrlDebug lon_ctrl_debug;
    ::hozon::netaos::MbdLatCtrlDebug lat_ctrl_debug;
    ::hozon::netaos::MbdCtrlOutputDebug ctrl_output_debug;
    ::hozon::netaos::VersionNumber_A VersionNumber;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::HafMbdDebug,pos_calc_debug,traj_calc_debug,adas_calc_debug,ctrl_dec_debug,lon_ctrl_debug,lat_ctrl_debug,ctrl_output_debug,VersionNumber);

#endif // HOZON_NETAOS_IMPL_TYPE_HAFMBDDEBUG_H_
/* EOF */
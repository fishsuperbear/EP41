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
 * @file impl_type_dtdebug_hafpartnrecusysfltinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTDEBUG_HAFPARTNRECUSYSFLTINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_DTDEBUG_HAFPARTNRECUSYSFLTINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtDebug_HafpartnrEcuSysFltInfo {
    std::uint8_t ESC_Sys_IDB_Flt;
    std::uint8_t EPB_Sys_Flt;
    std::uint8_t IDB_Sys_Flt;
    std::uint8_t ABS_Sys_IDB_Flt;
    std::uint8_t EBD_Sys_IDB_Flt;
    std::uint8_t TCS_Sys_IDB_Flt;
    std::uint8_t Str_A_Sys_Flt;
    std::uint8_t CDCS_Sys_Flt;
    std::uint8_t ESC_Sys_RCU7_Flt;
    std::uint8_t TCS_Sys_RCU_Flt;
    std::uint8_t HSA_Sys_RCU_Flt;
    std::uint8_t ABS_Sys_RCU_Flt;
    std::uint8_t ESC_Sys_RCU5_Flt;
    std::uint8_t BUCKLE_Sys_Flt;
    std::uint8_t EPS_ADAS_Sys_Flt;
    std::uint8_t EPS_PA_Sys_Flt;
    std::uint8_t WHEEL_rr_Sys_Flt;
    std::uint8_t WHEEL_lr_Sys_Flt;
    std::uint8_t WHEEL_rf_Sys_Flt;
    std::uint8_t WHEEL_lf_Sys_Flt;
    std::uint8_t IDB_PreFill_Sys_Flt;
    std::uint8_t IDB_PA_Sts_Sys_Flt;
    std::uint8_t IDB_Jerk_Sys_Flt;
    std::uint8_t ESC_Sys_Off_Flt;
    std::uint8_t HBA_Sys_Flt;
    std::uint8_t ROP_Sys_Flt;
    std::uint8_t HSA_Sys_Flt;
    std::uint8_t HDC_Sys_Flt;
    std::uint8_t GearPos_Sys_Flt;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtDebug_HafpartnrEcuSysFltInfo,ESC_Sys_IDB_Flt,EPB_Sys_Flt,IDB_Sys_Flt,ABS_Sys_IDB_Flt,EBD_Sys_IDB_Flt,TCS_Sys_IDB_Flt,Str_A_Sys_Flt,CDCS_Sys_Flt,ESC_Sys_RCU7_Flt,TCS_Sys_RCU_Flt,HSA_Sys_RCU_Flt,ABS_Sys_RCU_Flt,ESC_Sys_RCU5_Flt,BUCKLE_Sys_Flt,EPS_ADAS_Sys_Flt,EPS_PA_Sys_Flt,WHEEL_rr_Sys_Flt,WHEEL_lr_Sys_Flt,WHEEL_rf_Sys_Flt,WHEEL_lf_Sys_Flt,IDB_PreFill_Sys_Flt,IDB_PA_Sts_Sys_Flt,IDB_Jerk_Sys_Flt,ESC_Sys_Off_Flt,HBA_Sys_Flt,ROP_Sys_Flt,HSA_Sys_Flt,HDC_Sys_Flt,GearPos_Sys_Flt);

#endif // HOZON_NETAOS_IMPL_TYPE_DTDEBUG_HAFPARTNRECUSYSFLTINFO_H_
/* EOF */
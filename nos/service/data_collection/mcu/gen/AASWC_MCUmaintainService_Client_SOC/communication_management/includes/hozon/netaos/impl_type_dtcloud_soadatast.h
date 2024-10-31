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
 * @file impl_type_dtcloud_soadatast.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_SOADATAST_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_SOADATAST_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_soadatast_array_1005.h"
#include "hozon/netaos/impl_type_dtcloud_soadatast_array_1006.h"
#include "hozon/netaos/impl_type_dtcloud_soadatast_array_1007.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_SOADatast {
    std::uint32_t TrajData_locSeq;
    std::uint32_t TrajData_sec;
    std::uint32_t TrajData_nsec;
    std::uint16_t TrajData_utmZoneID;
    std::uint32_t PoseData_seq;
    std::uint32_t PoseData_sec;
    std::uint32_t PoseData_nsec;
    std::uint8_t PoseData_locationState;
    std::uint32_t SnsrFsnLaneDate_locSeq;
    std::uint32_t SnsrFsnLaneDate_sec;
    std::uint32_t SnsrFsnLaneDate_nsec;
    std::uint32_t SnsrFsnObj_locSeq;
    std::uint32_t SnsrFsnObj_sec;
    std::uint32_t SnsrFsnObj_nsec;
    std::uint8_t Lowpower_Rqevent;
    ::hozon::netaos::DtCloud_SOADatast_Array_1005 VehicleCfgF170_CfgTmp;
    ::hozon::netaos::DtCloud_SOADatast_Array_1006 Eth_30_Tc3xx_TxBufferState;
    ::hozon::netaos::DtCloud_SOADatast_Array_1007 EnabledOfRouteGrpSoConDyn;
    std::uint32_t Eth_Read_ARSseq;
    std::uint32_t Eth_Read_SRR_FLseq;
    std::uint32_t Eth_Read_SRR_FRseq;
    std::uint32_t Eth_Read_SRR_RLseq;
    std::uint32_t ImuSeq;
    std::uint32_t ChassisSeq;
    std::uint32_t GnssSeq;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_SOADatast,TrajData_locSeq,TrajData_sec,TrajData_nsec,TrajData_utmZoneID,PoseData_seq,PoseData_sec,PoseData_nsec,PoseData_locationState,SnsrFsnLaneDate_locSeq,SnsrFsnLaneDate_sec,SnsrFsnLaneDate_nsec,SnsrFsnObj_locSeq,SnsrFsnObj_sec,SnsrFsnObj_nsec,Lowpower_Rqevent,VehicleCfgF170_CfgTmp,Eth_30_Tc3xx_TxBufferState,EnabledOfRouteGrpSoConDyn,Eth_Read_ARSseq,Eth_Read_SRR_FLseq,Eth_Read_SRR_FRseq,Eth_Read_SRR_RLseq,ImuSeq,ChassisSeq,GnssSeq);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_SOADATAST_H_
/* EOF */
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
 * @file impl_type_dtcloud_nvm.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_NVM_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_NVM_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_ads2nvm_funmode.h"
#include "hozon/netaos/impl_type_dtcloud_ctrl_nvm_data.h"
#include "hozon/netaos/impl_type_dtcloud_egomemmsg.h"
#include "hozon/netaos/impl_type_dtcloud_hafaeb2nvm.h"
#include "hozon/netaos/impl_type_dtcloud_hafcan2nvmbus.h"
#include "hozon/netaos/impl_type_dtcloud_hafswc2aeb.h"
#include "hozon/netaos/impl_type_dtcloud_nvm2ads_mcuinputs.h"
#include "hozon/netaos/impl_type_dtcloud_nvmblockstate.h"
#include "hozon/netaos/impl_type_dtcloud_rb_nvm_asw_remmberstate.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_NVM {
    ::hozon::netaos::DtCloud_HafCAN2NVMBus CAN_NVM_SOCData;
    ::hozon::netaos::DtCloud_ADS2NVM_FunMode ADS_NVM_Data;
    ::hozon::netaos::DtCloud_Ctrl_NVM_Data Ctrl_NVM_Data;
    ::hozon::netaos::DtCloud_HafAEB2NVM AEB_NVM_Data;
    std::uint8_t VehicleCfgD44;
    std::uint8_t VehicleCfgD45;
    std::uint8_t VehicleCfgD47;
    std::uint8_t VehicleCfgD48;
    ::hozon::netaos::DtCloud_NVM2ADS_MCUInputs NVM_ADS_Data;
    ::hozon::netaos::DtCloud_Ctrl_NVM_Data NVM_Ctrl_Data;
    ::hozon::netaos::DtCloud_HafSwc2AEB NVM_AEB_Data;
    ::hozon::netaos::DtCloud_EgoMemMsg NVM_ETH_SOCData;
    ::hozon::netaos::DtCloud_NVMBlockState Nvm_Blcok_StateEth;
    ::hozon::netaos::DtCloud_rb_NvM_ASW_RemmberState NvM_ASW_RemmberState_Data;
    ::hozon::netaos::DtCloud_rb_NvM_ASW_RemmberState App_ASW_RemmberState_Data;
    std::uint32_t TaskCntNVMread_10ms;
    std::uint32_t TaskCntNVMwrite_10ms;
    std::uint32_t TaskCntNVMmainfunction_10ms;
    std::uint8_t EthCloud_WR_NVM_Count;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_NVM,CAN_NVM_SOCData,ADS_NVM_Data,Ctrl_NVM_Data,AEB_NVM_Data,VehicleCfgD44,VehicleCfgD45,VehicleCfgD47,VehicleCfgD48,NVM_ADS_Data,NVM_Ctrl_Data,NVM_AEB_Data,NVM_ETH_SOCData,Nvm_Blcok_StateEth,NvM_ASW_RemmberState_Data,App_ASW_RemmberState_Data,TaskCntNVMread_10ms,TaskCntNVMwrite_10ms,TaskCntNVMmainfunction_10ms,EthCloud_WR_NVM_Count);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_NVM_H_
/* EOF */
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
 * @file impl_type_dtcloud_sm.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_SM_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_SM_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_dsvpwrinfo.h"
#include "hozon/netaos/impl_type_dtcloud_dw_transpermission.h"
#include "hozon/netaos/impl_type_dtcloud_lifecycle_mgr_info.h"
#include "hozon/netaos/impl_type_dtcloud_modechange_overtiming.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_SM {
    ::hozon::netaos::DtCloud_DsvPwrInfo DsvIf_DsvPwrInfo;
    ::hozon::netaos::DtCloud_ModeChange_OverTiming PwrMgr_Timing;
    ::hozon::netaos::DtCloud_DW_TransPermission PwrMgr_TransPermission_DW;
    std::uint8_t g_DsvPwrCmd;
    std::uint8_t PwrMgr_State;
    std::uint8_t NM_State;
    std::uint8_t ThermM_State;
    std::uint8_t g_PMOrinMode;
    std::uint8_t PM_LIFECYCLE_STATE;
    std::uint8_t g_GuardM_PwrOnSOC;
    std::uint8_t g_SOCErrorPowerUpOrDown;
    std::uint8_t g_PwrMgrModeFaultStatus;
    std::uint8_t g_EthPmSocRequestState;
    std::uint8_t g_PMMcuRespondState;
    ::hozon::netaos::DtCloud_LifeCycle_Mgr_Info LifeCycle_Mgr_Info;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_SM,DsvIf_DsvPwrInfo,PwrMgr_Timing,PwrMgr_TransPermission_DW,g_DsvPwrCmd,PwrMgr_State,NM_State,ThermM_State,g_PMOrinMode,PM_LIFECYCLE_STATE,g_GuardM_PwrOnSOC,g_SOCErrorPowerUpOrDown,g_PwrMgrModeFaultStatus,g_EthPmSocRequestState,g_PMMcuRespondState,LifeCycle_Mgr_Info);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_SM_H_
/* EOF */
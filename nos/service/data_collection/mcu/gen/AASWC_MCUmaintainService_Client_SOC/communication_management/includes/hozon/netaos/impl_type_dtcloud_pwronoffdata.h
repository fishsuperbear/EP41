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
 * @file impl_type_dtcloud_pwronoffdata.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_PWRONOFFDATA_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_PWRONOFFDATA_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_pwronoffdata_array_1002.h"
#include "hozon/netaos/impl_type_dtcloud_pwronoffdata_array_1003.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_PwrOnOffData {
    std::uint8_t nvmreadflag;
    std::uint8_t PwrMgr_DelayCoolSended;
    ::hozon::netaos::DtCloud_PwrOnOffData_Array_1002 DsvIf_DsvPwrInfo;
    ::hozon::netaos::DtCloud_PwrOnOffData_Array_1003 PwrMgr_TransPermission_DW;
    std::uint8_t g_DsvPwrCmd;
    std::uint8_t PwrMgr_State;
    std::uint8_t g_NmRxFlag;
    std::uint8_t gComPowerStep;
    std::uint8_t ThermM_State;
    std::uint8_t Reset_Req;
    std::uint8_t g_PMOrinMode;
    std::uint8_t g_GuardM_PwrOnSOC;
    std::uint8_t g_PwrmgrUpGradeMode;
    std::uint8_t g_PwrmgrSocForceShutdown;
    std::uint8_t g_EthPmSocRequestState;
    std::uint8_t g_MCUL2State;
    std::uint32_t PwrMgrWriteTaskHour;
    std::uint32_t PwrMgrWriteTaskMinute;
    std::uint32_t NmBusSleepSecond;
    std::uint16_t year;
    std::uint8_t month;
    std::uint8_t day;
    std::uint8_t hour;
    std::uint8_t minute;
    std::uint8_t second;
    std::uint16_t msecond;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_PwrOnOffData,nvmreadflag,PwrMgr_DelayCoolSended,DsvIf_DsvPwrInfo,PwrMgr_TransPermission_DW,g_DsvPwrCmd,PwrMgr_State,g_NmRxFlag,gComPowerStep,ThermM_State,Reset_Req,g_PMOrinMode,g_GuardM_PwrOnSOC,g_PwrmgrUpGradeMode,g_PwrmgrSocForceShutdown,g_EthPmSocRequestState,g_MCUL2State,PwrMgrWriteTaskHour,PwrMgrWriteTaskMinute,NmBusSleepSecond,year,month,day,hour,minute,second,msecond);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_PWRONOFFDATA_H_
/* EOF */
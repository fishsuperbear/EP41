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
 * @file impl_type_dtdebug_fm.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTDEBUG_FM_H_
#define HOZON_NETAOS_IMPL_TYPE_DTDEBUG_FM_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtdebug_fm_array_1017.h"
#include "hozon/netaos/impl_type_dtdebug_fm_array_1018.h"
#include "hozon/netaos/impl_type_dtdebug_fm_array_1019.h"
#include "hozon/netaos/impl_type_dtdebug_hafaebfcwfaultbus.h"
#include "hozon/netaos/impl_type_dtdebug_haffltmgreventbusoutarray.h"
#include "hozon/netaos/impl_type_dtservcallfail.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtDebug_FM {
    std::uint32_t cntEntry;
    std::uint32_t CntFaultClustReport;
    std::uint8_t FltMgr_Module_State;
    ::hozon::netaos::DtDebug_FM_Array_1017 frtSyncEventWatchInfo;
    std::uint8_t SyncEventQueueSendIdx;
    std::uint16_t socFaultReportCnt;
    ::hozon::netaos::DtDebug_FM_Array_1018 FltMgr_MISC_RAW_LIST;
    ::hozon::netaos::DtDebug_FM_Array_1019 FltMgr_SOC_MISC_RAW_LIST;
    std::uint8_t curFaultMaxProcVal;
    ::hozon::netaos::DtDebug_HafFltMgrEventBusOutArray FltMgrEventBusInfo;
    std::uint8_t FrstSyncToSocFlg;
    std::uint8_t PwrModePostProcReqVal;
    std::uint8_t testCnt_FM_Main;
    ::hozon::netaos::DtDebug_HafAebFcwFaultBus FltMgr_HafAebFcwFaultBus;
    ::hozon::netaos::DtServCallFail FM_ETH_ServCallFail;
    ::hozon::netaos::DtServCallFail FM_PwrM_ServCallFail;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtDebug_FM,cntEntry,CntFaultClustReport,FltMgr_Module_State,frtSyncEventWatchInfo,SyncEventQueueSendIdx,socFaultReportCnt,FltMgr_MISC_RAW_LIST,FltMgr_SOC_MISC_RAW_LIST,curFaultMaxProcVal,FltMgrEventBusInfo,FrstSyncToSocFlg,PwrModePostProcReqVal,testCnt_FM_Main,FltMgr_HafAebFcwFaultBus,FM_ETH_ServCallFail,FM_PwrM_ServCallFail);

#endif // HOZON_NETAOS_IMPL_TYPE_DTDEBUG_FM_H_
/* EOF */
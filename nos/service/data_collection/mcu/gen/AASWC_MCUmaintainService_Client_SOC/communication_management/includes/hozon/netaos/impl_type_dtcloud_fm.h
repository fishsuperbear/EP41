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
 * @file impl_type_dtcloud_fm.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_FM_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_FM_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_hafaebfcwfaultbus.h"
#include "hozon/netaos/impl_type_dtcloud_haffltmgreventbusoutarray.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_FM {
    std::uint32_t cntEntry;
    std::uint32_t CntFaultClustReport;
    std::uint8_t FltMgr_Module_State;
    std::uint8_t SyncEventQueueSendIdx;
    std::uint16_t socFaultReportCnt;
    ::hozon::netaos::DtCloud_HafFltMgrEventBusOutArray FltMgrEventBusInfo;
    std::uint8_t FrstSyncToSocFlg;
    std::uint8_t PwrModePostProcReqVal;
    std::uint8_t testCnt_FM_Main;
    ::hozon::netaos::DtCloud_HafAebFcwFaultBus FltMgr_HafAebFcwFaultBus;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_FM,cntEntry,CntFaultClustReport,FltMgr_Module_State,SyncEventQueueSendIdx,socFaultReportCnt,FltMgrEventBusInfo,FrstSyncToSocFlg,PwrModePostProcReqVal,testCnt_FM_Main,FltMgr_HafAebFcwFaultBus);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_FM_H_
/* EOF */
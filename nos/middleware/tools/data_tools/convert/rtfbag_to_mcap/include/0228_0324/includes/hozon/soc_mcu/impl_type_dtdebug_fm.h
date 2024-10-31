/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_FM_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_FM_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "hozon/soc_mcu/impl_type_dtdebug_eventsenttoeth_100.h"
#include "impl_type_uint16.h"
#include "hozon/soc_mcu/impl_type_uint8array_83.h"
#include "hozon/soc_mcu/impl_type_uint8array_11.h"
#include "hozon/soc_mcu/impl_type_dtdebug_haffltmgreventbusoutarray.h"
#include "hozon/soc_mcu/impl_type_dtdebug_hafaebfcwfault.h"
#include "hozon/soc_mcu/impl_type_dtservcallfail.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_FM {
    ::UInt32 cntEntry;
    ::UInt32 CntFaultClustReport;
    ::UInt8 FltMgr_Module_State;
    ::hozon::soc_mcu::DtDebug_EventSentToETH_100 frtSyncEventWatchInfo;
    ::UInt8 SyncEventQueueSendIdx;
    ::UInt16 socFaultReportCnt;
    ::hozon::soc_mcu::uint8Array_83 FltMgr_MISC_RAW_LIST;
    ::hozon::soc_mcu::uint8Array_11 FltMgr_SOC_MISC_RAW_LIST;
    ::UInt8 curFaultMaxProcVal;
    ::hozon::soc_mcu::DtDebug_HafFltMgrEventBusOutArray FltMgrEventBusInfo;
    ::UInt8 FrstSyncToSocFlg;
    ::UInt8 PwrModePostProcReqVal;
    ::UInt8 testCnt_FM_Main;
    ::hozon::soc_mcu::DtDebug_HafAebFcwFault FltMgr_HafAebFcwFaultBus;
    ::hozon::soc_mcu::DtServCallFail FM_ETH_ServCallFail;
    ::hozon::soc_mcu::DtServCallFail FM_PwrM_ServCallFail;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(cntEntry);
        fun(CntFaultClustReport);
        fun(FltMgr_Module_State);
        fun(frtSyncEventWatchInfo);
        fun(SyncEventQueueSendIdx);
        fun(socFaultReportCnt);
        fun(FltMgr_MISC_RAW_LIST);
        fun(FltMgr_SOC_MISC_RAW_LIST);
        fun(curFaultMaxProcVal);
        fun(FltMgrEventBusInfo);
        fun(FrstSyncToSocFlg);
        fun(PwrModePostProcReqVal);
        fun(testCnt_FM_Main);
        fun(FltMgr_HafAebFcwFaultBus);
        fun(FM_ETH_ServCallFail);
        fun(FM_PwrM_ServCallFail);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(cntEntry);
        fun(CntFaultClustReport);
        fun(FltMgr_Module_State);
        fun(frtSyncEventWatchInfo);
        fun(SyncEventQueueSendIdx);
        fun(socFaultReportCnt);
        fun(FltMgr_MISC_RAW_LIST);
        fun(FltMgr_SOC_MISC_RAW_LIST);
        fun(curFaultMaxProcVal);
        fun(FltMgrEventBusInfo);
        fun(FrstSyncToSocFlg);
        fun(PwrModePostProcReqVal);
        fun(testCnt_FM_Main);
        fun(FltMgr_HafAebFcwFaultBus);
        fun(FM_ETH_ServCallFail);
        fun(FM_PwrM_ServCallFail);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("cntEntry", cntEntry);
        fun("CntFaultClustReport", CntFaultClustReport);
        fun("FltMgr_Module_State", FltMgr_Module_State);
        fun("frtSyncEventWatchInfo", frtSyncEventWatchInfo);
        fun("SyncEventQueueSendIdx", SyncEventQueueSendIdx);
        fun("socFaultReportCnt", socFaultReportCnt);
        fun("FltMgr_MISC_RAW_LIST", FltMgr_MISC_RAW_LIST);
        fun("FltMgr_SOC_MISC_RAW_LIST", FltMgr_SOC_MISC_RAW_LIST);
        fun("curFaultMaxProcVal", curFaultMaxProcVal);
        fun("FltMgrEventBusInfo", FltMgrEventBusInfo);
        fun("FrstSyncToSocFlg", FrstSyncToSocFlg);
        fun("PwrModePostProcReqVal", PwrModePostProcReqVal);
        fun("testCnt_FM_Main", testCnt_FM_Main);
        fun("FltMgr_HafAebFcwFaultBus", FltMgr_HafAebFcwFaultBus);
        fun("FM_ETH_ServCallFail", FM_ETH_ServCallFail);
        fun("FM_PwrM_ServCallFail", FM_PwrM_ServCallFail);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("cntEntry", cntEntry);
        fun("CntFaultClustReport", CntFaultClustReport);
        fun("FltMgr_Module_State", FltMgr_Module_State);
        fun("frtSyncEventWatchInfo", frtSyncEventWatchInfo);
        fun("SyncEventQueueSendIdx", SyncEventQueueSendIdx);
        fun("socFaultReportCnt", socFaultReportCnt);
        fun("FltMgr_MISC_RAW_LIST", FltMgr_MISC_RAW_LIST);
        fun("FltMgr_SOC_MISC_RAW_LIST", FltMgr_SOC_MISC_RAW_LIST);
        fun("curFaultMaxProcVal", curFaultMaxProcVal);
        fun("FltMgrEventBusInfo", FltMgrEventBusInfo);
        fun("FrstSyncToSocFlg", FrstSyncToSocFlg);
        fun("PwrModePostProcReqVal", PwrModePostProcReqVal);
        fun("testCnt_FM_Main", testCnt_FM_Main);
        fun("FltMgr_HafAebFcwFaultBus", FltMgr_HafAebFcwFaultBus);
        fun("FM_ETH_ServCallFail", FM_ETH_ServCallFail);
        fun("FM_PwrM_ServCallFail", FM_PwrM_ServCallFail);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_FM& t) const
    {
        return (cntEntry == t.cntEntry) && (CntFaultClustReport == t.CntFaultClustReport) && (FltMgr_Module_State == t.FltMgr_Module_State) && (frtSyncEventWatchInfo == t.frtSyncEventWatchInfo) && (SyncEventQueueSendIdx == t.SyncEventQueueSendIdx) && (socFaultReportCnt == t.socFaultReportCnt) && (FltMgr_MISC_RAW_LIST == t.FltMgr_MISC_RAW_LIST) && (FltMgr_SOC_MISC_RAW_LIST == t.FltMgr_SOC_MISC_RAW_LIST) && (curFaultMaxProcVal == t.curFaultMaxProcVal) && (FltMgrEventBusInfo == t.FltMgrEventBusInfo) && (FrstSyncToSocFlg == t.FrstSyncToSocFlg) && (PwrModePostProcReqVal == t.PwrModePostProcReqVal) && (testCnt_FM_Main == t.testCnt_FM_Main) && (FltMgr_HafAebFcwFaultBus == t.FltMgr_HafAebFcwFaultBus) && (FM_ETH_ServCallFail == t.FM_ETH_ServCallFail) && (FM_PwrM_ServCallFail == t.FM_PwrM_ServCallFail);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_FM_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_FM_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_FM_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "hozon/soc_mcu/impl_type_dtcloud_haffltmgreventbusoutarray.h"
#include "hozon/soc_mcu/impl_type_dtcloud_hafaebfcwfault.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_FM {
    ::UInt32 cntEntry;
    ::UInt32 CntFaultClustReport;
    ::UInt8 FltMgr_Module_State;
    ::UInt8 SyncEventQueueSendIdx;
    ::UInt16 socFaultReportCnt;
    ::hozon::soc_mcu::DtCloud_HafFltMgrEventBusOutArray FltMgrEventBusInfo;
    ::UInt8 FrstSyncToSocFlg;
    ::UInt8 PwrModePostProcReqVal;
    ::UInt8 testCnt_FM_Main;
    ::hozon::soc_mcu::DtCloud_HafAebFcwFault FltMgr_HafAebFcwFaultBus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(cntEntry);
        fun(CntFaultClustReport);
        fun(FltMgr_Module_State);
        fun(SyncEventQueueSendIdx);
        fun(socFaultReportCnt);
        fun(FltMgrEventBusInfo);
        fun(FrstSyncToSocFlg);
        fun(PwrModePostProcReqVal);
        fun(testCnt_FM_Main);
        fun(FltMgr_HafAebFcwFaultBus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(cntEntry);
        fun(CntFaultClustReport);
        fun(FltMgr_Module_State);
        fun(SyncEventQueueSendIdx);
        fun(socFaultReportCnt);
        fun(FltMgrEventBusInfo);
        fun(FrstSyncToSocFlg);
        fun(PwrModePostProcReqVal);
        fun(testCnt_FM_Main);
        fun(FltMgr_HafAebFcwFaultBus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("cntEntry", cntEntry);
        fun("CntFaultClustReport", CntFaultClustReport);
        fun("FltMgr_Module_State", FltMgr_Module_State);
        fun("SyncEventQueueSendIdx", SyncEventQueueSendIdx);
        fun("socFaultReportCnt", socFaultReportCnt);
        fun("FltMgrEventBusInfo", FltMgrEventBusInfo);
        fun("FrstSyncToSocFlg", FrstSyncToSocFlg);
        fun("PwrModePostProcReqVal", PwrModePostProcReqVal);
        fun("testCnt_FM_Main", testCnt_FM_Main);
        fun("FltMgr_HafAebFcwFaultBus", FltMgr_HafAebFcwFaultBus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("cntEntry", cntEntry);
        fun("CntFaultClustReport", CntFaultClustReport);
        fun("FltMgr_Module_State", FltMgr_Module_State);
        fun("SyncEventQueueSendIdx", SyncEventQueueSendIdx);
        fun("socFaultReportCnt", socFaultReportCnt);
        fun("FltMgrEventBusInfo", FltMgrEventBusInfo);
        fun("FrstSyncToSocFlg", FrstSyncToSocFlg);
        fun("PwrModePostProcReqVal", PwrModePostProcReqVal);
        fun("testCnt_FM_Main", testCnt_FM_Main);
        fun("FltMgr_HafAebFcwFaultBus", FltMgr_HafAebFcwFaultBus);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_FM& t) const
    {
        return (cntEntry == t.cntEntry) && (CntFaultClustReport == t.CntFaultClustReport) && (FltMgr_Module_State == t.FltMgr_Module_State) && (SyncEventQueueSendIdx == t.SyncEventQueueSendIdx) && (socFaultReportCnt == t.socFaultReportCnt) && (FltMgrEventBusInfo == t.FltMgrEventBusInfo) && (FrstSyncToSocFlg == t.FrstSyncToSocFlg) && (PwrModePostProcReqVal == t.PwrModePostProcReqVal) && (testCnt_FM_Main == t.testCnt_FM_Main) && (FltMgr_HafAebFcwFaultBus == t.FltMgr_HafAebFcwFaultBus);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_FM_H

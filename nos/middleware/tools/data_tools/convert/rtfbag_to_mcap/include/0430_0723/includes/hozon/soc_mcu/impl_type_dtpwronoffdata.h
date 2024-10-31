/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTPWRONOFFDATA_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTPWRONOFFDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint32.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace soc_mcu {
struct DtPwrOnOffData {
    ::UInt8 PwrMgr_DelayCoolSended;
    ::UInt8 PwrMgr_AccDisable;
    ::UInt8 HwSm_SetMDCPwrStatus_Debug;
    ::UInt8 curPwrMgrState;
    ::UInt8 NMSocWakeUpIndication;
    ::UInt8 FmSocForceShutDownIndication;
    ::UInt8 FmSocForceResetIndication;
    ::UInt8 FmMdcForceShutDownIndication;
    ::UInt8 FmMdcForceResetIndication;
    ::UInt32 Timing_ToStandByAndShutDown;
    ::UInt32 Timing_ToStandAlone;
    ::UInt32 Timing_ToWorking;
    ::UInt8 NMMDCDefaultIndication;
    ::UInt8 NmSocShutDownIndication;
    ::UInt8 NmBusSleepIndication;
    ::UInt8 gPwrMgr_HwSmRteStatus;
    ::UInt16 year;
    ::UInt8 month;
    ::UInt8 day;
    ::UInt8 hour;
    ::UInt8 minute;
    ::UInt8 second;
    ::UInt16 msecond;
    ::UInt8 TimingOutFlag;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(PwrMgr_DelayCoolSended);
        fun(PwrMgr_AccDisable);
        fun(HwSm_SetMDCPwrStatus_Debug);
        fun(curPwrMgrState);
        fun(NMSocWakeUpIndication);
        fun(FmSocForceShutDownIndication);
        fun(FmSocForceResetIndication);
        fun(FmMdcForceShutDownIndication);
        fun(FmMdcForceResetIndication);
        fun(Timing_ToStandByAndShutDown);
        fun(Timing_ToStandAlone);
        fun(Timing_ToWorking);
        fun(NMMDCDefaultIndication);
        fun(NmSocShutDownIndication);
        fun(NmBusSleepIndication);
        fun(gPwrMgr_HwSmRteStatus);
        fun(year);
        fun(month);
        fun(day);
        fun(hour);
        fun(minute);
        fun(second);
        fun(msecond);
        fun(TimingOutFlag);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(PwrMgr_DelayCoolSended);
        fun(PwrMgr_AccDisable);
        fun(HwSm_SetMDCPwrStatus_Debug);
        fun(curPwrMgrState);
        fun(NMSocWakeUpIndication);
        fun(FmSocForceShutDownIndication);
        fun(FmSocForceResetIndication);
        fun(FmMdcForceShutDownIndication);
        fun(FmMdcForceResetIndication);
        fun(Timing_ToStandByAndShutDown);
        fun(Timing_ToStandAlone);
        fun(Timing_ToWorking);
        fun(NMMDCDefaultIndication);
        fun(NmSocShutDownIndication);
        fun(NmBusSleepIndication);
        fun(gPwrMgr_HwSmRteStatus);
        fun(year);
        fun(month);
        fun(day);
        fun(hour);
        fun(minute);
        fun(second);
        fun(msecond);
        fun(TimingOutFlag);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("PwrMgr_DelayCoolSended", PwrMgr_DelayCoolSended);
        fun("PwrMgr_AccDisable", PwrMgr_AccDisable);
        fun("HwSm_SetMDCPwrStatus_Debug", HwSm_SetMDCPwrStatus_Debug);
        fun("curPwrMgrState", curPwrMgrState);
        fun("NMSocWakeUpIndication", NMSocWakeUpIndication);
        fun("FmSocForceShutDownIndication", FmSocForceShutDownIndication);
        fun("FmSocForceResetIndication", FmSocForceResetIndication);
        fun("FmMdcForceShutDownIndication", FmMdcForceShutDownIndication);
        fun("FmMdcForceResetIndication", FmMdcForceResetIndication);
        fun("Timing_ToStandByAndShutDown", Timing_ToStandByAndShutDown);
        fun("Timing_ToStandAlone", Timing_ToStandAlone);
        fun("Timing_ToWorking", Timing_ToWorking);
        fun("NMMDCDefaultIndication", NMMDCDefaultIndication);
        fun("NmSocShutDownIndication", NmSocShutDownIndication);
        fun("NmBusSleepIndication", NmBusSleepIndication);
        fun("gPwrMgr_HwSmRteStatus", gPwrMgr_HwSmRteStatus);
        fun("year", year);
        fun("month", month);
        fun("day", day);
        fun("hour", hour);
        fun("minute", minute);
        fun("second", second);
        fun("msecond", msecond);
        fun("TimingOutFlag", TimingOutFlag);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("PwrMgr_DelayCoolSended", PwrMgr_DelayCoolSended);
        fun("PwrMgr_AccDisable", PwrMgr_AccDisable);
        fun("HwSm_SetMDCPwrStatus_Debug", HwSm_SetMDCPwrStatus_Debug);
        fun("curPwrMgrState", curPwrMgrState);
        fun("NMSocWakeUpIndication", NMSocWakeUpIndication);
        fun("FmSocForceShutDownIndication", FmSocForceShutDownIndication);
        fun("FmSocForceResetIndication", FmSocForceResetIndication);
        fun("FmMdcForceShutDownIndication", FmMdcForceShutDownIndication);
        fun("FmMdcForceResetIndication", FmMdcForceResetIndication);
        fun("Timing_ToStandByAndShutDown", Timing_ToStandByAndShutDown);
        fun("Timing_ToStandAlone", Timing_ToStandAlone);
        fun("Timing_ToWorking", Timing_ToWorking);
        fun("NMMDCDefaultIndication", NMMDCDefaultIndication);
        fun("NmSocShutDownIndication", NmSocShutDownIndication);
        fun("NmBusSleepIndication", NmBusSleepIndication);
        fun("gPwrMgr_HwSmRteStatus", gPwrMgr_HwSmRteStatus);
        fun("year", year);
        fun("month", month);
        fun("day", day);
        fun("hour", hour);
        fun("minute", minute);
        fun("second", second);
        fun("msecond", msecond);
        fun("TimingOutFlag", TimingOutFlag);
    }

    bool operator==(const ::hozon::soc_mcu::DtPwrOnOffData& t) const
    {
        return (PwrMgr_DelayCoolSended == t.PwrMgr_DelayCoolSended) && (PwrMgr_AccDisable == t.PwrMgr_AccDisable) && (HwSm_SetMDCPwrStatus_Debug == t.HwSm_SetMDCPwrStatus_Debug) && (curPwrMgrState == t.curPwrMgrState) && (NMSocWakeUpIndication == t.NMSocWakeUpIndication) && (FmSocForceShutDownIndication == t.FmSocForceShutDownIndication) && (FmSocForceResetIndication == t.FmSocForceResetIndication) && (FmMdcForceShutDownIndication == t.FmMdcForceShutDownIndication) && (FmMdcForceResetIndication == t.FmMdcForceResetIndication) && (Timing_ToStandByAndShutDown == t.Timing_ToStandByAndShutDown) && (Timing_ToStandAlone == t.Timing_ToStandAlone) && (Timing_ToWorking == t.Timing_ToWorking) && (NMMDCDefaultIndication == t.NMMDCDefaultIndication) && (NmSocShutDownIndication == t.NmSocShutDownIndication) && (NmBusSleepIndication == t.NmBusSleepIndication) && (gPwrMgr_HwSmRteStatus == t.gPwrMgr_HwSmRteStatus) && (year == t.year) && (month == t.month) && (day == t.day) && (hour == t.hour) && (minute == t.minute) && (second == t.second) && (msecond == t.msecond) && (TimingOutFlag == t.TimingOutFlag);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTPWRONOFFDATA_H

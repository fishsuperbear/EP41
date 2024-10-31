/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_CLIENTSERVICERUNTIME_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_CLIENTSERVICERUNTIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_Sd_ClientServiceRuntime {
    ::UInt8 SoTcpConOpend;
    ::UInt16 SoConOpend;
    ::UInt8 RetryEnabled;
    ::UInt16 ServicePhase;
    ::UInt16 LastServicePhase;
    ::UInt16 State;
    ::UInt8 CurRepetition;
    ::UInt16 TxFindCount;
    ::UInt16 RxOfferCount;
    ::UInt16 RxStopOfferCount;
    ::UInt16 TxSubscribeCount;
    ::UInt16 TxStopSubscribeCount;
    ::UInt16 RxSubscribeAckCount;
    ::UInt16 RxSubscribeNackCount;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SoTcpConOpend);
        fun(SoConOpend);
        fun(RetryEnabled);
        fun(ServicePhase);
        fun(LastServicePhase);
        fun(State);
        fun(CurRepetition);
        fun(TxFindCount);
        fun(RxOfferCount);
        fun(RxStopOfferCount);
        fun(TxSubscribeCount);
        fun(TxStopSubscribeCount);
        fun(RxSubscribeAckCount);
        fun(RxSubscribeNackCount);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SoTcpConOpend);
        fun(SoConOpend);
        fun(RetryEnabled);
        fun(ServicePhase);
        fun(LastServicePhase);
        fun(State);
        fun(CurRepetition);
        fun(TxFindCount);
        fun(RxOfferCount);
        fun(RxStopOfferCount);
        fun(TxSubscribeCount);
        fun(TxStopSubscribeCount);
        fun(RxSubscribeAckCount);
        fun(RxSubscribeNackCount);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SoTcpConOpend", SoTcpConOpend);
        fun("SoConOpend", SoConOpend);
        fun("RetryEnabled", RetryEnabled);
        fun("ServicePhase", ServicePhase);
        fun("LastServicePhase", LastServicePhase);
        fun("State", State);
        fun("CurRepetition", CurRepetition);
        fun("TxFindCount", TxFindCount);
        fun("RxOfferCount", RxOfferCount);
        fun("RxStopOfferCount", RxStopOfferCount);
        fun("TxSubscribeCount", TxSubscribeCount);
        fun("TxStopSubscribeCount", TxStopSubscribeCount);
        fun("RxSubscribeAckCount", RxSubscribeAckCount);
        fun("RxSubscribeNackCount", RxSubscribeNackCount);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SoTcpConOpend", SoTcpConOpend);
        fun("SoConOpend", SoConOpend);
        fun("RetryEnabled", RetryEnabled);
        fun("ServicePhase", ServicePhase);
        fun("LastServicePhase", LastServicePhase);
        fun("State", State);
        fun("CurRepetition", CurRepetition);
        fun("TxFindCount", TxFindCount);
        fun("RxOfferCount", RxOfferCount);
        fun("RxStopOfferCount", RxStopOfferCount);
        fun("TxSubscribeCount", TxSubscribeCount);
        fun("TxStopSubscribeCount", TxStopSubscribeCount);
        fun("RxSubscribeAckCount", RxSubscribeAckCount);
        fun("RxSubscribeNackCount", RxSubscribeNackCount);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_Sd_ClientServiceRuntime& t) const
    {
        return (SoTcpConOpend == t.SoTcpConOpend) && (SoConOpend == t.SoConOpend) && (RetryEnabled == t.RetryEnabled) && (ServicePhase == t.ServicePhase) && (LastServicePhase == t.LastServicePhase) && (State == t.State) && (CurRepetition == t.CurRepetition) && (TxFindCount == t.TxFindCount) && (RxOfferCount == t.RxOfferCount) && (RxStopOfferCount == t.RxStopOfferCount) && (TxSubscribeCount == t.TxSubscribeCount) && (TxStopSubscribeCount == t.TxStopSubscribeCount) && (RxSubscribeAckCount == t.RxSubscribeAckCount) && (RxSubscribeNackCount == t.RxSubscribeNackCount);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_CLIENTSERVICERUNTIME_H

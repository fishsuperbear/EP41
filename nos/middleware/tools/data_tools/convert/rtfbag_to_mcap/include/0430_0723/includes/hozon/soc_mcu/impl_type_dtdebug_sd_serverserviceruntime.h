/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_SERVERSERVICERUNTIME_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_SERVERSERVICERUNTIME_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_Sd_ServerServiceRuntime {
    ::UInt8 SoConOpened;
    ::UInt16 RxFindCount;
    ::UInt16 TxOfferCount;
    ::UInt16 TxStopOfferCount;
    ::UInt16 RxSubscribeCount;
    ::UInt16 RxStopSubscribeCount;
    ::UInt16 TxSubscribeAckCount;
    ::UInt16 TxSubscribeNackCount;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SoConOpened);
        fun(RxFindCount);
        fun(TxOfferCount);
        fun(TxStopOfferCount);
        fun(RxSubscribeCount);
        fun(RxStopSubscribeCount);
        fun(TxSubscribeAckCount);
        fun(TxSubscribeNackCount);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SoConOpened);
        fun(RxFindCount);
        fun(TxOfferCount);
        fun(TxStopOfferCount);
        fun(RxSubscribeCount);
        fun(RxStopSubscribeCount);
        fun(TxSubscribeAckCount);
        fun(TxSubscribeNackCount);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SoConOpened", SoConOpened);
        fun("RxFindCount", RxFindCount);
        fun("TxOfferCount", TxOfferCount);
        fun("TxStopOfferCount", TxStopOfferCount);
        fun("RxSubscribeCount", RxSubscribeCount);
        fun("RxStopSubscribeCount", RxStopSubscribeCount);
        fun("TxSubscribeAckCount", TxSubscribeAckCount);
        fun("TxSubscribeNackCount", TxSubscribeNackCount);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SoConOpened", SoConOpened);
        fun("RxFindCount", RxFindCount);
        fun("TxOfferCount", TxOfferCount);
        fun("TxStopOfferCount", TxStopOfferCount);
        fun("RxSubscribeCount", RxSubscribeCount);
        fun("RxStopSubscribeCount", RxStopSubscribeCount);
        fun("TxSubscribeAckCount", TxSubscribeAckCount);
        fun("TxSubscribeNackCount", TxSubscribeNackCount);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_Sd_ServerServiceRuntime& t) const
    {
        return (SoConOpened == t.SoConOpened) && (RxFindCount == t.RxFindCount) && (TxOfferCount == t.TxOfferCount) && (TxStopOfferCount == t.TxStopOfferCount) && (RxSubscribeCount == t.RxSubscribeCount) && (RxStopSubscribeCount == t.RxStopSubscribeCount) && (TxSubscribeAckCount == t.TxSubscribeAckCount) && (TxSubscribeNackCount == t.TxSubscribeNackCount);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SD_SERVERSERVICERUNTIME_H

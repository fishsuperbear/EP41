/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_ETH_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_ETH_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtdebug_soadata.h"
#include "hozon/soc_mcu/impl_type_dtdebug_sd_serverservice_19.h"
#include "hozon/soc_mcu/impl_type_dtdebug_sd_clientservice_12.h"
#include "hozon/soc_mcu/impl_type_dtdebug_sd_serverserviceruntime_19.h"
#include "hozon/soc_mcu/impl_type_dtdebug_sd_clientserviceruntime_12.h"
#include "hozon/soc_mcu/impl_type_dtdebug_someiptp_rxdata_4.h"
#include "hozon/soc_mcu/impl_type_dtdebug_someiptp_txdata_2.h"
#include "hozon/soc_mcu/impl_type_dtdebug_counterseq_24.h"
#include "hozon/soc_mcu/impl_type_dtdebug_wsthafglobaltime.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_ETH {
    ::hozon::soc_mcu::DtDebug_SOAData SOAData;
    ::hozon::soc_mcu::DtDebug_Sd_ServerService_19 Sd_ServerServicehz;
    ::hozon::soc_mcu::DtDebug_Sd_ClientService_12 Sd_ClientServicehz;
    ::hozon::soc_mcu::DtDebug_Sd_ServerServiceRuntime_19 Sd_ServerServiceRuntimehz;
    ::hozon::soc_mcu::DtDebug_Sd_ClientServiceRuntime_12 Sd_ClientServiceRuntimehz;
    ::hozon::soc_mcu::DtDebug_SomeIpTp_RxData_4 SomeIpTp_RxDatahz;
    ::hozon::soc_mcu::DtDebug_SomeIpTp_TxData_2 SomeIpTp_TxDatahz;
    ::hozon::soc_mcu::DtDebug_counterseq_24 counterseqhz;
    ::hozon::soc_mcu::DtDebug_wstHafGlobalTime HafGlobalTime;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SOAData);
        fun(Sd_ServerServicehz);
        fun(Sd_ClientServicehz);
        fun(Sd_ServerServiceRuntimehz);
        fun(Sd_ClientServiceRuntimehz);
        fun(SomeIpTp_RxDatahz);
        fun(SomeIpTp_TxDatahz);
        fun(counterseqhz);
        fun(HafGlobalTime);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SOAData);
        fun(Sd_ServerServicehz);
        fun(Sd_ClientServicehz);
        fun(Sd_ServerServiceRuntimehz);
        fun(Sd_ClientServiceRuntimehz);
        fun(SomeIpTp_RxDatahz);
        fun(SomeIpTp_TxDatahz);
        fun(counterseqhz);
        fun(HafGlobalTime);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SOAData", SOAData);
        fun("Sd_ServerServicehz", Sd_ServerServicehz);
        fun("Sd_ClientServicehz", Sd_ClientServicehz);
        fun("Sd_ServerServiceRuntimehz", Sd_ServerServiceRuntimehz);
        fun("Sd_ClientServiceRuntimehz", Sd_ClientServiceRuntimehz);
        fun("SomeIpTp_RxDatahz", SomeIpTp_RxDatahz);
        fun("SomeIpTp_TxDatahz", SomeIpTp_TxDatahz);
        fun("counterseqhz", counterseqhz);
        fun("HafGlobalTime", HafGlobalTime);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SOAData", SOAData);
        fun("Sd_ServerServicehz", Sd_ServerServicehz);
        fun("Sd_ClientServicehz", Sd_ClientServicehz);
        fun("Sd_ServerServiceRuntimehz", Sd_ServerServiceRuntimehz);
        fun("Sd_ClientServiceRuntimehz", Sd_ClientServiceRuntimehz);
        fun("SomeIpTp_RxDatahz", SomeIpTp_RxDatahz);
        fun("SomeIpTp_TxDatahz", SomeIpTp_TxDatahz);
        fun("counterseqhz", counterseqhz);
        fun("HafGlobalTime", HafGlobalTime);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_ETH& t) const
    {
        return (SOAData == t.SOAData) && (Sd_ServerServicehz == t.Sd_ServerServicehz) && (Sd_ClientServicehz == t.Sd_ClientServicehz) && (Sd_ServerServiceRuntimehz == t.Sd_ServerServiceRuntimehz) && (Sd_ClientServiceRuntimehz == t.Sd_ClientServiceRuntimehz) && (SomeIpTp_RxDatahz == t.SomeIpTp_RxDatahz) && (SomeIpTp_TxDatahz == t.SomeIpTp_TxDatahz) && (counterseqhz == t.counterseqhz) && (HafGlobalTime == t.HafGlobalTime);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_ETH_H

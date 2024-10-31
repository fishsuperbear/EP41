/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SENSORS_IMPL_TYPE_UPA_INFO_T_H
#define HOZON_SENSORS_IMPL_TYPE_UPA_INFO_T_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint32.h"
#include "hozon/sensors/impl_type_tsonarinfo.h"
#include "impl_type_uint16.h"
#include "hozon/composite/impl_type_uint16arry_3.h"

namespace hozon {
namespace sensors {
struct UPA_Info_T {
    ::UInt32 dwTimeStampH;
    ::UInt32 dwTimeStampL;
    ::hozon::sensors::tSonarInfo TxSnsInfo;
    ::hozon::sensors::tSonarInfo RxSns0Info;
    ::hozon::sensors::tSonarInfo RxSns1Info;
    ::hozon::sensors::tSonarInfo RxSns2Info;
    ::UInt16 wTxSns_Ringtime;
    ::hozon::composite::uint16Arry_3 wTxSns_Echo_Dist;
    ::hozon::composite::uint16Arry_3 wRxSns0_Echo_Dist;
    ::hozon::composite::uint16Arry_3 wRxSns1_Echo_Dist;
    ::hozon::composite::uint16Arry_3 wRxSns2_Echo_Dist;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(dwTimeStampH);
        fun(dwTimeStampL);
        fun(TxSnsInfo);
        fun(RxSns0Info);
        fun(RxSns1Info);
        fun(RxSns2Info);
        fun(wTxSns_Ringtime);
        fun(wTxSns_Echo_Dist);
        fun(wRxSns0_Echo_Dist);
        fun(wRxSns1_Echo_Dist);
        fun(wRxSns2_Echo_Dist);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(dwTimeStampH);
        fun(dwTimeStampL);
        fun(TxSnsInfo);
        fun(RxSns0Info);
        fun(RxSns1Info);
        fun(RxSns2Info);
        fun(wTxSns_Ringtime);
        fun(wTxSns_Echo_Dist);
        fun(wRxSns0_Echo_Dist);
        fun(wRxSns1_Echo_Dist);
        fun(wRxSns2_Echo_Dist);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("dwTimeStampH", dwTimeStampH);
        fun("dwTimeStampL", dwTimeStampL);
        fun("TxSnsInfo", TxSnsInfo);
        fun("RxSns0Info", RxSns0Info);
        fun("RxSns1Info", RxSns1Info);
        fun("RxSns2Info", RxSns2Info);
        fun("wTxSns_Ringtime", wTxSns_Ringtime);
        fun("wTxSns_Echo_Dist", wTxSns_Echo_Dist);
        fun("wRxSns0_Echo_Dist", wRxSns0_Echo_Dist);
        fun("wRxSns1_Echo_Dist", wRxSns1_Echo_Dist);
        fun("wRxSns2_Echo_Dist", wRxSns2_Echo_Dist);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("dwTimeStampH", dwTimeStampH);
        fun("dwTimeStampL", dwTimeStampL);
        fun("TxSnsInfo", TxSnsInfo);
        fun("RxSns0Info", RxSns0Info);
        fun("RxSns1Info", RxSns1Info);
        fun("RxSns2Info", RxSns2Info);
        fun("wTxSns_Ringtime", wTxSns_Ringtime);
        fun("wTxSns_Echo_Dist", wTxSns_Echo_Dist);
        fun("wRxSns0_Echo_Dist", wRxSns0_Echo_Dist);
        fun("wRxSns1_Echo_Dist", wRxSns1_Echo_Dist);
        fun("wRxSns2_Echo_Dist", wRxSns2_Echo_Dist);
    }

    bool operator==(const ::hozon::sensors::UPA_Info_T& t) const
    {
        return (dwTimeStampH == t.dwTimeStampH) && (dwTimeStampL == t.dwTimeStampL) && (TxSnsInfo == t.TxSnsInfo) && (RxSns0Info == t.RxSns0Info) && (RxSns1Info == t.RxSns1Info) && (RxSns2Info == t.RxSns2Info) && (wTxSns_Ringtime == t.wTxSns_Ringtime) && (wTxSns_Echo_Dist == t.wTxSns_Echo_Dist) && (wRxSns0_Echo_Dist == t.wRxSns0_Echo_Dist) && (wRxSns1_Echo_Dist == t.wRxSns1_Echo_Dist) && (wRxSns2_Echo_Dist == t.wRxSns2_Echo_Dist);
    }
};
} // namespace sensors
} // namespace hozon


#endif // HOZON_SENSORS_IMPL_TYPE_UPA_INFO_T_H

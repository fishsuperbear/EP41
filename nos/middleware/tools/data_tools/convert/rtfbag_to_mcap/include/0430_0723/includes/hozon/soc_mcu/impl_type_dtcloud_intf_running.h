/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_RUNNING_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_RUNNING_H
#include <cfloat>
#include <cmath>
#include "hozon/soc_mcu/impl_type_dtcloud_intf_runnablestatus.h"
#include "hozon/soc_mcu/impl_type_dtcloud_intf_datatransmitstatus.h"
#include "hozon/soc_mcu/impl_type_dtcloud_intf_signalsendstatus.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_INTF_Running {
    ::hozon::soc_mcu::DtCloud_INTF_RunnableStatus INTF_RunnableStatus;
    ::hozon::soc_mcu::DtCloud_INTF_DataTransmitStatus INTF_DataTransmitStatus;
    ::hozon::soc_mcu::DtCloud_INTF_SignalSendStatus INTF_SignalSendStatus;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(INTF_RunnableStatus);
        fun(INTF_DataTransmitStatus);
        fun(INTF_SignalSendStatus);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(INTF_RunnableStatus);
        fun(INTF_DataTransmitStatus);
        fun(INTF_SignalSendStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("INTF_RunnableStatus", INTF_RunnableStatus);
        fun("INTF_DataTransmitStatus", INTF_DataTransmitStatus);
        fun("INTF_SignalSendStatus", INTF_SignalSendStatus);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("INTF_RunnableStatus", INTF_RunnableStatus);
        fun("INTF_DataTransmitStatus", INTF_DataTransmitStatus);
        fun("INTF_SignalSendStatus", INTF_SignalSendStatus);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_INTF_Running& t) const
    {
        return (INTF_RunnableStatus == t.INTF_RunnableStatus) && (INTF_DataTransmitStatus == t.INTF_DataTransmitStatus) && (INTF_SignalSendStatus == t.INTF_SignalSendStatus);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_INTF_RUNNING_H

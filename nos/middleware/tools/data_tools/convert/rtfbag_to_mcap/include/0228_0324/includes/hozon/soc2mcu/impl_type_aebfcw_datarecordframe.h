/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC2MCU_IMPL_TYPE_AEBFCW_DATARECORDFRAME_H
#define HOZON_SOC2MCU_IMPL_TYPE_AEBFCW_DATARECORDFRAME_H
#include <cfloat>
#include <cmath>
#include "hozon/soc2mcu/impl_type_hafcan3bus_rx.h"
#include "hozon/soc2mcu/impl_type_hafswc2aebtype.h"
#include "hozon/soc2mcu/impl_type_hafaebfcwfaultbustype.h"
#include "hozon/soc2mcu/impl_type_hafcan3bus_tx.h"
#include "hozon/soc2mcu/impl_type_hafaeb2swctype.h"
#include "hozon/soc2mcu/impl_type_hafaeb2nvmtype.h"
#include "hozon/soc2mcu/impl_type_hafaebsdebugbustype.h"

namespace hozon {
namespace soc2mcu {
struct AEBFCW_DataRecordFrame {
    ::hozon::soc2mcu::HafCan3Bus_Rx Pi_AEBCan3BusRx;
    ::hozon::soc2mcu::HafSwc2AEBType HafSwc2AEBTypeData;
    ::hozon::soc2mcu::HafAebFcwFaultBusType HafAebFcwFaultBus;
    ::hozon::soc2mcu::HafCan3Bus_Tx Pi_AEBCan3BusTx;
    ::hozon::soc2mcu::HafAEB2SwcType HafAEB2SwcTypeData;
    ::hozon::soc2mcu::HafAEB2NVMType HafAEB2NVMTypeData;
    ::hozon::soc2mcu::HafAEBSDebugBusType HafAEBSDebugBusTypeData;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(Pi_AEBCan3BusRx);
        fun(HafSwc2AEBTypeData);
        fun(HafAebFcwFaultBus);
        fun(Pi_AEBCan3BusTx);
        fun(HafAEB2SwcTypeData);
        fun(HafAEB2NVMTypeData);
        fun(HafAEBSDebugBusTypeData);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(Pi_AEBCan3BusRx);
        fun(HafSwc2AEBTypeData);
        fun(HafAebFcwFaultBus);
        fun(Pi_AEBCan3BusTx);
        fun(HafAEB2SwcTypeData);
        fun(HafAEB2NVMTypeData);
        fun(HafAEBSDebugBusTypeData);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("Pi_AEBCan3BusRx", Pi_AEBCan3BusRx);
        fun("HafSwc2AEBTypeData", HafSwc2AEBTypeData);
        fun("HafAebFcwFaultBus", HafAebFcwFaultBus);
        fun("Pi_AEBCan3BusTx", Pi_AEBCan3BusTx);
        fun("HafAEB2SwcTypeData", HafAEB2SwcTypeData);
        fun("HafAEB2NVMTypeData", HafAEB2NVMTypeData);
        fun("HafAEBSDebugBusTypeData", HafAEBSDebugBusTypeData);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("Pi_AEBCan3BusRx", Pi_AEBCan3BusRx);
        fun("HafSwc2AEBTypeData", HafSwc2AEBTypeData);
        fun("HafAebFcwFaultBus", HafAebFcwFaultBus);
        fun("Pi_AEBCan3BusTx", Pi_AEBCan3BusTx);
        fun("HafAEB2SwcTypeData", HafAEB2SwcTypeData);
        fun("HafAEB2NVMTypeData", HafAEB2NVMTypeData);
        fun("HafAEBSDebugBusTypeData", HafAEBSDebugBusTypeData);
    }

    bool operator==(const ::hozon::soc2mcu::AEBFCW_DataRecordFrame& t) const
    {
        return (Pi_AEBCan3BusRx == t.Pi_AEBCan3BusRx) && (HafSwc2AEBTypeData == t.HafSwc2AEBTypeData) && (HafAebFcwFaultBus == t.HafAebFcwFaultBus) && (Pi_AEBCan3BusTx == t.Pi_AEBCan3BusTx) && (HafAEB2SwcTypeData == t.HafAEB2SwcTypeData) && (HafAEB2NVMTypeData == t.HafAEB2NVMTypeData) && (HafAEBSDebugBusTypeData == t.HafAEBSDebugBusTypeData);
    }
};
} // namespace soc2mcu
} // namespace hozon


#endif // HOZON_SOC2MCU_IMPL_TYPE_AEBFCW_DATARECORDFRAME_H

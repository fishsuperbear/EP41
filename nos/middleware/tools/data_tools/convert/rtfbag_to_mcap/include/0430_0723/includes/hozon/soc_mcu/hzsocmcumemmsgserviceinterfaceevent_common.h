/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_HZSOCMCUMEMMSGSERVICEINTERFACEEVENT_COMMON_H
#define HOZON_SOC_MCU_HZSOCMCUMEMMSGSERVICEINTERFACEEVENT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/chassis/impl_type_algmcutoegoframeeth.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace soc_mcu {

class HzSocMcuMemMsgServiceInterfaceEvent {
public:
    constexpr HzSocMcuMemMsgServiceInterfaceEvent() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HzSocMcuMemMsgServiceInterfaceEvent");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace soc_mcu
} // namespace hozon

#endif // HOZON_SOC_MCU_HZSOCMCUMEMMSGSERVICEINTERFACEEVENT_COMMON_H

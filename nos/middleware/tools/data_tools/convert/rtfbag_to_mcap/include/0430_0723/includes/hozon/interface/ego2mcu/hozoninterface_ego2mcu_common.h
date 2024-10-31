/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_EGO2MCU_HOZONINTERFACE_EGO2MCU_COMMON_H
#define HOZON_INTERFACE_EGO2MCU_HOZONINTERFACE_EGO2MCU_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/chassis/impl_type_algegotomcuframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace ego2mcu {

class HozonInterface_Ego2Mcu {
public:
    constexpr HozonInterface_Ego2Mcu() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Ego2Mcu");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace ego2mcu
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_EGO2MCU_HOZONINTERFACE_EGO2MCU_COMMON_H

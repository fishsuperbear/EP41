/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_MCU2EGO_HOZONINTERFACE_MCU2EGO_COMMON_H
#define HOZON_INTERFACE_MCU2EGO_HOZONINTERFACE_MCU2EGO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/chassis/impl_type_algmcutoegoframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace mcu2ego {

class HozonInterface_Mcu2Ego {
public:
    constexpr HozonInterface_Mcu2Ego() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Mcu2Ego");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace mcu2ego
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_MCU2EGO_HOZONINTERFACE_MCU2EGO_COMMON_H

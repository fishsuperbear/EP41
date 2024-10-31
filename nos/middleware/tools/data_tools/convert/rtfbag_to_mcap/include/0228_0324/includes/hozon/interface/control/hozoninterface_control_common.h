/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_CONTROL_HOZONINTERFACE_CONTROL_COMMON_H
#define HOZON_INTERFACE_CONTROL_HOZONINTERFACE_CONTROL_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/control/impl_type_controlframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace control {

class HozonInterface_Control {
public:
    constexpr HozonInterface_Control() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Control");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace control
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_CONTROL_HOZONINTERFACE_CONTROL_COMMON_H

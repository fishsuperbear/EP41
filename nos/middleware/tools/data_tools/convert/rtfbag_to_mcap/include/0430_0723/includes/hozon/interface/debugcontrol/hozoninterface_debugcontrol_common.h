/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DEBUGCONTROL_HOZONINTERFACE_DEBUGCONTROL_COMMON_H
#define HOZON_INTERFACE_DEBUGCONTROL_HOZONINTERFACE_DEBUGCONTROL_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/debugcontrol/impl_type_debugcontrolframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace debugcontrol {

class HozonInterface_DebugControl {
public:
    constexpr HozonInterface_DebugControl() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_DebugControl");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace debugcontrol
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DEBUGCONTROL_HOZONINTERFACE_DEBUGCONTROL_COMMON_H

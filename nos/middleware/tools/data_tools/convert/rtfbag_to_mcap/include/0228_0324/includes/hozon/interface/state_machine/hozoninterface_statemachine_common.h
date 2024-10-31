/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_STATEMACHINE_COMMON_H
#define HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_STATEMACHINE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/statemachine/impl_type_statemachineframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace state_machine {

class HozonInterface_StateMachine {
public:
    constexpr HozonInterface_StateMachine() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_StateMachine");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace state_machine
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_STATEMACHINE_COMMON_H

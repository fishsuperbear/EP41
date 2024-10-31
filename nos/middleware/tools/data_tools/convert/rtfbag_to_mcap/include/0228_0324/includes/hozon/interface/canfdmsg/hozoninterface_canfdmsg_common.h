/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_CANFDMSG_HOZONINTERFACE_CANFDMSG_COMMON_H
#define HOZON_INTERFACE_CANFDMSG_HOZONINTERFACE_CANFDMSG_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/canfdmsg/impl_type_canfdmsgframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace canfdmsg {

class HozonInterface_CanFdMsg {
public:
    constexpr HozonInterface_CanFdMsg() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_CanFdMsg");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace canfdmsg
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_CANFDMSG_HOZONINTERFACE_CANFDMSG_COMMON_H

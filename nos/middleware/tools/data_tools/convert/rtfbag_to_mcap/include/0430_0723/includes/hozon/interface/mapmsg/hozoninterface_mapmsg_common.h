/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_MAPMSG_HOZONINTERFACE_MAPMSG_COMMON_H
#define HOZON_INTERFACE_MAPMSG_HOZONINTERFACE_MAPMSG_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/mapmsg/impl_type_mapmsgframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace mapmsg {

class HozonInterface_MapMsg {
public:
    constexpr HozonInterface_MapMsg() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_MapMsg");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace mapmsg
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_MAPMSG_HOZONINTERFACE_MAPMSG_COMMON_H

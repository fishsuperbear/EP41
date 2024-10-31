/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_EGOHMI_HOZONINTERFACE_EGOHMI_COMMON_H
#define HOZON_INTERFACE_EGOHMI_HOZONINTERFACE_EGOHMI_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/chassis/impl_type_algegohmiframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace egohmi {

class HozonInterface_EgoHmi {
public:
    constexpr HozonInterface_EgoHmi() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_EgoHmi");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace egohmi
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_EGOHMI_HOZONINTERFACE_EGOHMI_COMMON_H

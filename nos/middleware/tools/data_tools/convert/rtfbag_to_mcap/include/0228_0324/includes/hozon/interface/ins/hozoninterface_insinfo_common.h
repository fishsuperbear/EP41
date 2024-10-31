/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_INS_HOZONINTERFACE_INSINFO_COMMON_H
#define HOZON_INTERFACE_INS_HOZONINTERFACE_INSINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_insinfoframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace ins {

class HozonInterface_InsInfo {
public:
    constexpr HozonInterface_InsInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_InsInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace ins
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_INS_HOZONINTERFACE_INSINFO_COMMON_H

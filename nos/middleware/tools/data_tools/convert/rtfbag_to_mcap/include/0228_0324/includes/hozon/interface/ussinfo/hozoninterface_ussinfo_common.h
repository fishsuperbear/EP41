/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_USSINFO_HOZONINTERFACE_USSINFO_COMMON_H
#define HOZON_INTERFACE_USSINFO_HOZONINTERFACE_USSINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_ussinfo.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace ussinfo {

class HozonInterface_UssInfo {
public:
    constexpr HozonInterface_UssInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_UssInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace ussinfo
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_USSINFO_HOZONINTERFACE_USSINFO_COMMON_H

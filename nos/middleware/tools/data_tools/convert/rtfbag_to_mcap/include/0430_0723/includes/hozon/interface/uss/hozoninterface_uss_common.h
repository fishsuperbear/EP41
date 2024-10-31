/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_USS_HOZONINTERFACE_USS_COMMON_H
#define HOZON_INTERFACE_USS_HOZONINTERFACE_USS_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_ussrawdataset.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace uss {

class HozonInterface_Uss {
public:
    constexpr HozonInterface_Uss() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_Uss");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace uss
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_USS_HOZONINTERFACE_USS_COMMON_H

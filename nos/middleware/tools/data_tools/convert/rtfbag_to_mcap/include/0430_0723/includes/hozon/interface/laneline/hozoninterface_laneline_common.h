/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LANELINE_HOZONINTERFACE_LANELINE_COMMON_H
#define HOZON_INTERFACE_LANELINE_HOZONINTERFACE_LANELINE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/laneline/impl_type_lanelineframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace laneline {

class HozonInterface_LaneLine {
public:
    constexpr HozonInterface_LaneLine() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_LaneLine");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace laneline
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LANELINE_HOZONINTERFACE_LANELINE_COMMON_H

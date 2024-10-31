/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_FREESPACE_HOZONINTERFACE_FREESPACE_COMMON_H
#define HOZON_INTERFACE_FREESPACE_HOZONINTERFACE_FREESPACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/freespace/impl_type_freespaceframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace freespace {

class HozonInterface_Freespace {
public:
    constexpr HozonInterface_Freespace() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_Freespace");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace freespace
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_FREESPACE_HOZONINTERFACE_FREESPACE_COMMON_H

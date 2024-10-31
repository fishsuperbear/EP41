/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DEBUGPB_HOZONINTERFACE_DEBUGPB_COMMON_H
#define HOZON_INTERFACE_DEBUGPB_HOZONINTERFACE_DEBUGPB_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/debugpb/impl_type_debugpbframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace debugpb {

class HozonInterface_DebugPb {
public:
    constexpr HozonInterface_DebugPb() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_DebugPb");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace debugpb
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DEBUGPB_HOZONINTERFACE_DEBUGPB_COMMON_H

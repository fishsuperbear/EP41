/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_PERCEPTIONINFO_HOZONINTERFACE_PERCEPTIONINFO_COMMON_H
#define HOZON_INTERFACE_PERCEPTIONINFO_HOZONINTERFACE_PERCEPTIONINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_algperceptioninfoframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace perceptioninfo {

class HozonInterface_PerceptionInfo {
public:
    constexpr HozonInterface_PerceptionInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_PerceptionInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace perceptioninfo
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_PERCEPTIONINFO_HOZONINTERFACE_PERCEPTIONINFO_COMMON_H

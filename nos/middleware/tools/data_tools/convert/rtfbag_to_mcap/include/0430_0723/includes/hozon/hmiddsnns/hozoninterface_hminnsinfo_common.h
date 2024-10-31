/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMIDDSNNS_HOZONINTERFACE_HMINNSINFO_COMMON_H
#define HOZON_HMIDDSNNS_HOZONINTERFACE_HMINNSINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/hmi/impl_type_nns_info_struct.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmiddsnns {

class HozonInterface_HmiNnsInfo {
public:
    constexpr HozonInterface_HmiNnsInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_HmiNnsInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace hmiddsnns
} // namespace hozon

#endif // HOZON_HMIDDSNNS_HOZONINTERFACE_HMINNSINFO_COMMON_H

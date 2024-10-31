/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMIDDSINS_HOZONINTERFACE_HMIINSINFO_COMMON_H
#define HOZON_HMIDDSINS_HOZONINTERFACE_HMIINSINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/hmi/impl_type_ins_info_struct.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmiddsins {

class HozonInterface_HmiInsInfo {
public:
    constexpr HozonInterface_HmiInsInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HozonInterface_HmiInsInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace hmiddsins
} // namespace hozon

#endif // HOZON_HMIDDSINS_HOZONINTERFACE_HMIINSINFO_COMMON_H

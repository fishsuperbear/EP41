/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LPM_SOCPOWERDDSSERVICEINTERFACE_COMMON_H
#define HOZON_LPM_SOCPOWERDDSSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_int8.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace lpm {

class SocPowerDdsServiceInterface {
public:
    constexpr SocPowerDdsServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/SocPowerDdsServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace lpm
} // namespace hozon

#endif // HOZON_LPM_SOCPOWERDDSSERVICEINTERFACE_COMMON_H

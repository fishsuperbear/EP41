/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SAMM_SAMMFEATURESERVICEINTERFACE_COMMON_H
#define ARA_SAMM_SAMMFEATURESERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/samm/impl_type_sammfeature.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace samm {

class SammFeatureServiceInterface {
public:
    constexpr SammFeatureServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/SammFeatureServiceInterface/SammFeatureServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace samm
} // namespace ara

#endif // ARA_SAMM_SAMMFEATURESERVICEINTERFACE_COMMON_H

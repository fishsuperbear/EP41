/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ADSFI_SCENEFEATUREINTF_COMMON_H
#define ADSFI_SCENEFEATUREINTF_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "adsfi/impl_type_scenefeature.h"
#include <cfloat>
#include <cmath>

namespace adsfi {

class SceneFeatureIntf {
public:
    constexpr SceneFeatureIntf() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/FunctionalSoftware/SceneFeatureServiceInterface/SceneFeatureIntf");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace adsfi

#endif // ADSFI_SCENEFEATUREINTF_COMMON_H

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_STCAMERA_STCAMERASERVICEINTERFACE_COMMON_H
#define HOZON_STCAMERA_STCAMERASERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/stcamera/impl_type_stcameradata.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace stcamera {

class StCameraServiceInterface {
public:
    constexpr StCameraServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/StCameraServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace stcamera
} // namespace hozon

#endif // HOZON_STCAMERA_STCAMERASERVICEINTERFACE_COMMON_H

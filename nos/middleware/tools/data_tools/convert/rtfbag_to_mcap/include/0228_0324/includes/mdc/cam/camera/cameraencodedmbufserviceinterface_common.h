/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_CAM_CAMERA_CAMERAENCODEDMBUFSERVICEINTERFACE_COMMON_H
#define MDC_CAM_CAMERA_CAMERAENCODEDMBUFSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/camera/impl_type_cameraencodedmbufstruct.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace cam {
namespace camera {

class CameraEncodedMbufServiceInterface {
public:
    constexpr CameraEncodedMbufServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace camera
} // namespace cam
} // namespace mdc

#endif // MDC_CAM_CAMERA_CAMERAENCODEDMBUFSERVICEINTERFACE_COMMON_H

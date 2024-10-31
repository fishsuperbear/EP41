/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_CAM_CAMERA_CAMERAINTERNALDATAMETHODSERVICEINTERFACE_COMMON_H
#define MDC_CAM_CAMERA_CAMERAINTERNALDATAMETHODSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_int32.h"
#include "ara/camera/impl_type_camerainternaldata.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace cam {
namespace camera {
namespace methods {
namespace GetCameraInternalData {
struct Output {
    ::ara::camera::CameraInternalData data;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(data);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(data);
    }

    bool operator==(const Output& t) const
    {
       return (data == t.data);
    }
};
} // namespace GetCameraInternalData
} // namespace methods

class CameraInternalDataMethodServiceInterface {
public:
    constexpr CameraInternalDataMethodServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/CameraInternalDataMethodServiceInterface/CameraInternalDataMethodServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace camera
} // namespace cam
} // namespace mdc

#endif // MDC_CAM_CAMERA_CAMERAINTERNALDATAMETHODSERVICEINTERFACE_COMMON_H

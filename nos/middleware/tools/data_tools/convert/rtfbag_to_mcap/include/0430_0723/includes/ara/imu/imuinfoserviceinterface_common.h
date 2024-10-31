/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_IMU_IMUINFOSERVICEINTERFACE_COMMON_H
#define ARA_IMU_IMUINFOSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "ara/imu/impl_type_imuinfo.h"
#include <cfloat>
#include <cmath>

namespace ara {
namespace imu {

class ImuInfoServiceInterface {
public:
    constexpr ImuInfoServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/ImuInfoServiceInterface/ImuInfoServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace imu
} // namespace ara

#endif // ARA_IMU_IMUINFOSERVICEINTERFACE_COMMON_H

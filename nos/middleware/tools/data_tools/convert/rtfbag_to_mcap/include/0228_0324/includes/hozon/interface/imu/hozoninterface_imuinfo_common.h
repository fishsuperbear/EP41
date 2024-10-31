/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_IMU_HOZONINTERFACE_IMUINFO_COMMON_H
#define HOZON_INTERFACE_IMU_HOZONINTERFACE_IMUINFO_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/sensors/impl_type_imuinsinfoframe.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace imu {

class HozonInterface_ImuInfo {
public:
    constexpr HozonInterface_ImuInfo() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_ImuInfo");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace imu
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_IMU_HOZONINTERFACE_IMUINFO_COMMON_H

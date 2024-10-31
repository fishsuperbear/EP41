/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_COMMON_H
#define HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/hmi/impl_type_adas_dataproperties_struct.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace hmi {

class HmiADASdataServiceInterface_0x0403 {
public:
    constexpr HmiADASdataServiceInterface_0x0403() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HmiADASdataServiceInterface_0x0403");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace hmi
} // namespace hozon

#endif // HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_COMMON_H

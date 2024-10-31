/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SM_HZSMTASKPROCSERVICEINTERFACE_COMMON_H
#define HOZON_SM_HZSMTASKPROCSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_string.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace sm {

class HzSmTaskProcServiceInterface {
public:
    constexpr HzSmTaskProcServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/hozon/ServiceInterface/HzSmTaskProcServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace sm
} // namespace hozon

#endif // HOZON_SM_HZSMTASKPROCSERVICEINTERFACE_COMMON_H

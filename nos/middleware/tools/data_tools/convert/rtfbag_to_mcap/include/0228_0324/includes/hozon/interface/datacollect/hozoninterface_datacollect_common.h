/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_COMMON_H
#define HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/datacollect/impl_type_collecttrigger.h"
#include "hozon/datacollect/impl_type_customcollectdata.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace datacollect {
namespace methods {
} // namespace methods

class HozonInterface_DataCollect {
public:
    constexpr HozonInterface_DataCollect() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_DataCollect");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace datacollect
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_COMMON_H

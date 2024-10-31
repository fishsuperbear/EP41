/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUDATACOLLECT_COMMON_H
#define HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUDATACOLLECT_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "hozon/soc_mcu/impl_type_struct_soc_mcu_array_algo.h"
#include "hozon/soc_mcu/impl_type_struct_soc_mcu_array_prj.h"
#include <cfloat>
#include <cmath>

namespace hozon {
namespace interface {
namespace datacollect {

class HozonInterface_MCUDataCollect {
public:
    constexpr HozonInterface_MCUDataCollect() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/Hozon/ServiceInterface/HozonInterface_MCUDataCollect");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace datacollect
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_MCUDATACOLLECT_COMMON_H
